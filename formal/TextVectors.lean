import KadmiumFormal.Native.ModelText

def main (args : List String) : IO Unit := do
  let path := args.headD "../crates/kadmium-runtime/tests/fixtures/model_text_contract.txt"
  let mut lines : Array String := #[]
  for maximum in (List.range 12).drop 1 do
    for count in List.range 15 do
      let accepted := (Kadmium.Native.Text.accept maximum (List.replicate count 7)).isSome
      lines := lines.push s!"context {maximum} {count} {accepted}"
  for token in List.range 8 do
    lines := lines.push s!"eos {token} {Kadmium.Native.Text.isEos [2, 4] token}"
  IO.FS.writeFile path (String.intercalate "\n" lines.toList ++ "\n")
