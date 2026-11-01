/-!
# `.ktb` bank layout

Models the page layout shared by `KtbBankPacker::pack_to_bytes` in `crates/kadmium-data/src/lib.rs`,
`write_bank` in `crates/kadmium-data/src/checkpoint.rs`, and the layout that `Checkpoint::verify`
recomputes:

```rust
let padded = bytes.max(1).checked_add(PAGE - 1)? / PAGE * PAGE;   // reservation
// page 0 holds the header; tensor i starts at PAGE + Σ_{j<i} padded_j and is copied byte for
// byte, then zero-padded to its reservation.
```

and the acceptance check of `KtbBankReader::tensor_range`: `PAGE ≤ start`,
`start + len ≤ archive`, `start % PAGE == 0`.

This is the storage half of the lossless-checkpoint contract (`AGENTS.md` section 1.1): a
normalised checkpoint is the source tensors copied into banks without a cast. The theorems say
where each tensor lands and that nothing overlaps or is lost:

* `offsets_aligned`, `PAGE_le_offsets`: every tensor starts on a page boundary at or after page 1.
* `offsets_disjoint`, `offsets_in_bounds`: tensor `i` ends at or before tensor `j > i` begins, and
  before the end of the bank. The reservations tile `[PAGE, total)`.
* `bank_getElem?`: byte `j` of tensor `i` is at `offsets[i] + j` of the bank image, unchanged. The
  copy is byte-exact for every tensor, including empty ones, which still reserve a page.
* `reader_accepts`: every packed entry passes `tensor_range`, so `from_slice(pack(..))` never
  rejects what the packer wrote.

Modelling choices: sizes and offsets are unbounded `Nat`. The Rust uses `checked_add` on the
reservation and the running offset and fails with "size overflow" instead of wrapping; the model
covers every call that does not fail. Page 0 is zero in the model and holds the JSON header in
Rust; `bank_getElem?` only speaks about offsets at or after `PAGE`.
-/

namespace Kadmium.Data

/-- `KTB_PAGE_ALIGNMENT_BYTES`. -/
def PAGE : Nat := 65536

/-- `padded_reservation_bytes`: the payload, at least one byte, rounded up to a page. -/
def padded (bytes : Nat) : Nat := (max bytes 1 + (PAGE - 1)) / PAGE * PAGE

theorem PAGE_pos : 0 < PAGE := by decide

theorem le_padded (b : Nat) : b ≤ padded b := by
  unfold padded
  have h1 := Nat.div_add_mod (max b 1 + (PAGE - 1)) PAGE
  have h2 := Nat.mod_lt (max b 1 + (PAGE - 1)) PAGE_pos
  have h3 : b ≤ max b 1 := Nat.le_max_left _ _
  rw [Nat.mul_comm]
  omega

theorem PAGE_le_padded (b : Nat) : PAGE ≤ padded b := by
  show PAGE ≤ (max b 1 + (PAGE - 1)) / PAGE * PAGE
  have h3 : 1 ≤ max b 1 := Nat.le_max_right _ _
  have hq : 1 ≤ (max b 1 + (PAGE - 1)) / PAGE :=
    (Nat.le_div_iff_mul_le PAGE_pos).mpr (by unfold PAGE at *; omega)
  have := Nat.mul_le_mul_right PAGE hq
  simpa using this

theorem PAGE_dvd_padded (b : Nat) : PAGE ∣ padded b := Nat.dvd_mul_left _ _

/-- A reservation never wastes a whole page. -/
theorem padded_lt (b : Nat) : padded b < max b 1 + PAGE := by
  unfold padded
  have := Nat.div_mul_le_self (max b 1 + (PAGE - 1)) PAGE
  have := PAGE_pos
  omega

theorem padded_zero : padded 0 = PAGE := by decide

/-! ## Offsets -/

/-- Start offset of every tensor, given the sizes in bank order and the first free offset. -/
def offsetsFrom (base : Nat) : List Nat → List Nat
  | [] => []
  | b :: rest => base :: offsetsFrom (base + padded b) rest

/-- `page_offset_bytes` of each entry: page 0 is the header. -/
def offsets (sizes : List Nat) : List Nat := offsetsFrom PAGE sizes

/-- The bank length: the header page plus every reservation. -/
def total (sizes : List Nat) : Nat := PAGE + (sizes.map padded).sum

@[simp] theorem length_offsetsFrom (base : Nat) (sizes : List Nat) :
    (offsetsFrom base sizes).length = sizes.length := by
  induction sizes generalizing base with
  | nil => rfl
  | cons b rest ih => simp [offsetsFrom, ih]

@[simp] theorem length_offsets (sizes : List Nat) : (offsets sizes).length = sizes.length :=
  length_offsetsFrom _ _

theorem offsetsFrom_getElem (base : Nat) (sizes : List Nat) (i : Nat) (h : i < sizes.length) :
    (offsetsFrom base sizes)[i]'(by simpa using h) = base + ((sizes.take i).map padded).sum := by
  induction sizes generalizing base i with
  | nil => simp at h
  | cons b rest ih =>
    cases i with
    | zero => simp [offsetsFrom]
    | succ i =>
      simp only [offsetsFrom, List.getElem_cons_succ, List.take_succ_cons, List.map_cons,
        List.sum_cons]
      rw [ih (base + padded b) i (by simpa using h)]
      omega

theorem offsets_getElem (sizes : List Nat) (i : Nat) (h : i < sizes.length) :
    (offsets sizes)[i]'(by simpa using h) = PAGE + ((sizes.take i).map padded).sum :=
  offsetsFrom_getElem PAGE sizes i h

theorem offsetsFrom_shift (base : Nat) (sizes : List Nat) (i : Nat) (h : i < sizes.length) :
    (offsetsFrom base sizes)[i]'(by simpa using h) =
      base + (offsetsFrom 0 sizes)[i]'(by simpa using h) := by
  rw [offsetsFrom_getElem _ _ _ h, offsetsFrom_getElem _ _ _ h]
  omega

theorem PAGE_dvd_sum_padded (l : List Nat) : PAGE ∣ (l.map padded).sum := by
  induction l with
  | nil => simp
  | cons b rest ih =>
    simp only [List.map_cons, List.sum_cons]
    exact Nat.dvd_add (PAGE_dvd_padded b) ih

/-- Every tensor starts on a page boundary. -/
theorem offsets_aligned (sizes : List Nat) (i : Nat) (h : i < sizes.length) :
    PAGE ∣ (offsets sizes)[i]'(by simpa using h) := by
  rw [offsets_getElem sizes i h]
  exact Nat.dvd_add (Nat.dvd_refl _) (PAGE_dvd_sum_padded _)

/-- Every tensor starts after the header page. -/
theorem PAGE_le_offsets (sizes : List Nat) (i : Nat) (h : i < sizes.length) :
    PAGE ≤ (offsets sizes)[i]'(by simpa using h) := by
  rw [offsets_getElem sizes i h]
  exact Nat.le_add_right _ _

theorem sum_map_take_le (l : List Nat) (f : Nat → Nat) {i j : Nat} (hij : i ≤ j) :
    ((l.take i).map f).sum ≤ ((l.take j).map f).sum := by
  have h1 : l.take i = (l.take j).take i := by rw [List.take_take, Nat.min_eq_left hij]
  have h2 := congrArg (fun t : List Nat => (t.map f).sum) (List.take_append_drop i (l.take j))
  simp only [List.map_append, List.sum_append] at h2
  rw [h1]
  omega

theorem sum_map_take_succ (l : List Nat) (f : Nat → Nat) (i : Nat) (h : i < l.length) :
    ((l.take (i + 1)).map f).sum = ((l.take i).map f).sum + f l[i] := by
  rw [List.take_succ_eq_append_getElem h, List.map_append, List.sum_append]
  simp

/-- Tensor `i` ends before tensor `j > i` begins: reservations never overlap. -/
theorem offsets_disjoint (sizes : List Nat) {i j : Nat} (hij : i < j) (hj : j < sizes.length) :
    (offsets sizes)[i]'(by simp; omega) + sizes[i]'(by omega) ≤ (offsets sizes)[j]'(by simpa using hj) := by
  rw [offsets_getElem sizes i (by omega), offsets_getElem sizes j hj]
  have h1 := sum_map_take_succ sizes padded i (by omega)
  have h2 := sum_map_take_le sizes padded (i := i + 1) (j := j) hij
  have h3 := le_padded (sizes[i]'(by omega))
  omega

/-- Every tensor ends inside the bank. -/
theorem offsets_in_bounds (sizes : List Nat) (i : Nat) (h : i < sizes.length) :
    (offsets sizes)[i]'(by simpa using h) + sizes[i] ≤ total sizes := by
  rw [offsets_getElem sizes i h]
  unfold total
  have h1 := sum_map_take_succ sizes padded i h
  have h2 := sum_map_take_le sizes padded (i := i + 1) (j := sizes.length) h
  rw [List.take_length] at h2
  have h3 := le_padded sizes[i]
  omega

/-! ## Bytes -/

/-- One tensor as written: its bytes then zero padding up to its reservation. -/
def pad (d : List Nat) : List Nat := d ++ List.replicate (padded d.length - d.length) 0

/-- The bank image after the header page. -/
def rel (data : List (List Nat)) : List Nat := (data.map pad).flatten

/-- The bank image, with page 0 zeroed (the Rust writes the JSON header there). -/
def bank (data : List (List Nat)) : List Nat := List.replicate PAGE 0 ++ rel data

theorem length_pad (d : List Nat) : (pad d).length = padded d.length := by
  simp only [pad, List.length_append, List.length_replicate]
  have := le_padded d.length
  omega

theorem length_rel (data : List (List Nat)) :
    (rel data).length = ((data.map List.length).map padded).sum := by
  induction data with
  | nil => rfl
  | cons d rest ih =>
    simp only [rel, List.map_cons, List.flatten_cons, List.length_append, length_pad, List.sum_cons]
    rw [← ih]
    rfl

theorem length_bank (data : List (List Nat)) :
    (bank data).length = total (data.map List.length) := by
  simp only [bank, List.length_append, List.length_replicate, length_rel, total]

theorem rel_getElem? (data : List (List Nat)) :
    ∀ (i j : Nat) (hi : i < data.length) (hj : j < (data[i]'hi).length),
      (rel data)[(offsetsFrom 0 (data.map List.length))[i]'(by simpa using hi) + j]? =
        some ((data[i]'hi)[j]'hj) := by
  induction data with
  | nil => intro i _ hi; simp at hi
  | cons d rest ih =>
    intro i j hi hj
    cases i with
    | zero =>
      simp only [List.map_cons, offsetsFrom, List.getElem_cons_zero, Nat.zero_add, rel,
        List.flatten_cons]
      have hjd : j < d.length := hj
      have hjp : j < (pad d).length := by rw [length_pad]; exact Nat.lt_of_lt_of_le hjd (le_padded _)
      rw [List.getElem?_append_left hjp]
      simp only [pad]
      rw [List.getElem?_append_left hjd]
      simp [hjd]
    | succ i =>
      simp only [List.map_cons, offsetsFrom, List.getElem_cons_succ, rel, List.flatten_cons]
      have hi' : i < rest.length := by simpa using hi
      rw [offsetsFrom_shift _ _ _ (by simpa using hi')]
      have hge : (pad d).length ≤ 0 + padded d.length +
          (offsetsFrom 0 (rest.map List.length))[i]'(by simpa using hi') + j := by
        rw [length_pad]; omega
      rw [List.getElem?_append_right hge, length_pad]
      rw [show 0 + padded d.length + (offsetsFrom 0 (rest.map List.length))[i]'(by simpa using hi')
        + j - padded d.length = (offsetsFrom 0 (rest.map List.length))[i]'(by simpa using hi') + j
        by omega]
      exact ih i j hi' hj

/-- **Byte-exact copy.** Byte `j` of tensor `i` sits at `offsets[i] + j` in the bank, unchanged. -/
theorem bank_getElem? (data : List (List Nat)) (i j : Nat) (hi : i < data.length)
    (hj : j < (data[i]'hi).length) :
    (bank data)[(offsets (data.map List.length))[i]'(by simpa using hi) + j]? =
      some ((data[i]'hi)[j]'hj) := by
  unfold bank offsets
  rw [offsetsFrom_shift _ _ _ (by simpa using hi)]
  have hge : (List.replicate PAGE (0 : Nat)).length ≤
      PAGE + (offsetsFrom 0 (data.map List.length))[i]'(by simpa using hi) + j := by
    rw [List.length_replicate]; omega
  rw [List.getElem?_append_right hge, List.length_replicate]
  rw [show PAGE + (offsetsFrom 0 (data.map List.length))[i]'(by simpa using hi) + j - PAGE =
    (offsetsFrom 0 (data.map List.length))[i]'(by simpa using hi) + j by omega]
  exact rel_getElem? data i j hi hj

/-! ## The reader's check -/

/-- `KtbBankReader::tensor_range` on an archive of `archive` bytes. -/
def rangeOk (archive start len : Nat) : Bool :=
  PAGE ≤ start && start + len ≤ archive && start % PAGE == 0

/-- Every entry the packer writes passes the reader's range check. -/
theorem reader_accepts (sizes : List Nat) (i : Nat) (h : i < sizes.length) :
    rangeOk (total sizes) ((offsets sizes)[i]'(by simpa using h)) sizes[i] = true := by
  simp only [rangeOk, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
  exact ⟨⟨PAGE_le_offsets sizes i h, offsets_in_bounds sizes i h⟩,
    Nat.mod_eq_zero_of_dvd (offsets_aligned sizes i h)⟩

end Kadmium.Data
