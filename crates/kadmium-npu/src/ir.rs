//! Intel NGRAPH_LITE payload: compiler version, two length-prefixed buffers
//! (OpenVINO IR XML and weights). Generated here without an OpenVINO dependency.
use crate::MatMulShape;

pub(crate) fn matmul(shape: MatMulShape, weights: &[f32], version: (u16, u16)) -> Vec<u8> {
    let (m, k, n) = (shape.m(), shape.k(), shape.n());
    let bytes = weights.len() * 4;
    let xml = format!(
        r#"<?xml version="1.0"?>
<net name="kadmium_matmul" version="11">
<layers>
<layer id="0" name="input" type="Parameter" version="opset1">
<data shape="{m},{k}" element_type="f32"/>
<output><port id="0" precision="FP32" names="input"><dim>{m}</dim><dim>{k}</dim></port></output>
</layer>
<layer id="1" name="weights" type="Const" version="opset1">
<data element_type="f32" shape="{k},{n}" offset="0" size="{bytes}"/>
<output><port id="0" precision="FP32"><dim>{k}</dim><dim>{n}</dim></port></output>
</layer>
<layer id="2" name="matmul" type="MatMul" version="opset1">
<data transpose_a="false" transpose_b="false"/>
<input><port id="0" precision="FP32"><dim>{m}</dim><dim>{k}</dim></port>
<port id="1" precision="FP32"><dim>{k}</dim><dim>{n}</dim></port></input>
<output><port id="2" precision="FP32" names="output"><dim>{m}</dim><dim>{n}</dim></port></output>
</layer>
<layer id="3" name="output" type="Result" version="opset1">
<input><port id="0" precision="FP32"><dim>{m}</dim><dim>{n}</dim></port></input>
</layer>
</layers>
<edges><edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/></edges>
</net>"#
    );
    let mut data = Vec::with_capacity(24 + xml.len() + bytes);
    data.extend_from_slice(&version.0.to_le_bytes());
    data.extend_from_slice(&version.1.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&(xml.len() as u64).to_le_bytes());
    data.extend_from_slice(xml.as_bytes());
    data.extend_from_slice(&(bytes as u64).to_le_bytes());
    for value in weights {
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wire_payload_has_version_xml_and_little_endian_weights() {
        let data = matmul(MatMulShape::new(2, 3, 1).unwrap(), &[1., -2., 0.5], (5, 9));
        assert_eq!(&data[..8], &[5, 0, 9, 0, 2, 0, 0, 0]);
        let len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        let xml = std::str::from_utf8(&data[16..16 + len]).unwrap();
        assert!(xml.contains("shape=\"2,3\""));
        assert!(xml.contains("shape=\"3,1\" offset=\"0\" size=\"12\""));
        assert_eq!(
            u64::from_le_bytes(data[16 + len..24 + len].try_into().unwrap()),
            12
        );
        assert_eq!(
            &data[24 + len..],
            &[1f32, -2., 0.5]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }
}
