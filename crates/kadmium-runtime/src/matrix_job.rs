//! Bounded, user-supplied FP32 matrix products on the existing TCP work protocol.
use crate::{ExecutorContext, WorkExecutor, WorkInputs, WorkOutput, repops};
use kadmium_grid::{AppKind, WorkUnit};
use kadmium_sharding::StepGraphBuilder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixProduct {
    pub rows: usize,
    pub inner: usize,
    pub columns: usize,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl MatrixProduct {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=256).contains(&self.rows)
            || !(1..=256).contains(&self.inner)
            || !(1..=256).contains(&self.columns)
        {
            return Err("Matrix dimensions must each be between 1 and 256".into());
        }
        if self.left.len() != self.rows * self.inner
            || self.right.len() != self.inner * self.columns
        {
            return Err("Matrix payload lengths do not match the row-major dimensions".into());
        }
        if self.left.iter().chain(&self.right).any(|v| !v.is_finite()) {
            return Err("Matrix values must be finite FP32 numbers".into());
        }
        Ok(())
    }

    pub fn multiply(&self) -> Result<Vec<f32>, String> {
        self.validate()?;
        let mut output = vec![0.; self.rows * self.columns];
        let mut column = vec![0.; self.inner];
        for c in 0..self.columns {
            for (k, value) in column.iter_mut().enumerate() {
                *value = self.right[k * self.columns + c];
            }
            for r in 0..self.rows {
                let value =
                    repops::tree_dot_f32(&self.left[r * self.inner..(r + 1) * self.inner], &column);
                if !value.is_finite() {
                    return Err(
                        "Matrix result overflowed FP32; no partial result was returned".into(),
                    );
                }
                output[r * self.columns + c] = value;
            }
        }
        Ok(output)
    }
}

pub struct MatrixProductExecutor;
impl WorkExecutor for MatrixProductExecutor {
    fn app_kind(&self) -> AppKind {
        AppKind::Custom("matrix-product-f32-v1".into())
    }
    fn execute(
        &self,
        unit: &WorkUnit,
        _inputs: &WorkInputs,
        _ctx: &ExecutorContext,
    ) -> Result<WorkOutput, String> {
        let params: MatrixProduct = unit.params_json().map_err(|e| e.to_string())?;
        let values = params.multiply()?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(WorkOutput {
            digest_fnv1a: repops::digest_bytes("matrix-product-f32-v1", &bytes),
            outputs: vec![(
                StepGraphBuilder::output_ref(
                    unit.graph_id.as_deref().unwrap_or("matrix"),
                    &unit.id,
                ),
                bytes,
            )],
            units_processed: params.rows as u64,
            ..WorkOutput::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_partition_preserves_full_precision_and_reduction_order() {
        let matrix = MatrixProduct {
            rows: 3,
            inner: 3,
            columns: 2,
            left: vec![1., 2., 3., -7., 0.125, 4., 1e10, -1e10, 0.25],
            right: vec![2., -3., 0.125, 4., 8., 1.],
        };
        let full = matrix.multiply().unwrap();
        assert_eq!(&full[..2], &[26.25, 8.]);
        let split: Vec<_> = matrix
            .left
            .chunks(3)
            .flat_map(|row| {
                MatrixProduct {
                    rows: 1,
                    left: row.to_vec(),
                    ..matrix.clone()
                }
                .multiply()
                .unwrap()
            })
            .collect();
        assert_eq!(
            full.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            split.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(
            MatrixProduct {
                rows: usize::MAX,
                ..matrix.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            MatrixProduct {
                left: vec![f32::NAN; 9],
                ..matrix
            }
            .multiply()
            .is_err()
        );
    }
}
