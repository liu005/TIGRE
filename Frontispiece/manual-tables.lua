-- Assign widths so long table cells wrap in the PDF.
function Table(tbl)
  local count = #tbl.colspecs
  if count == 0 then return tbl end
  for index, spec in ipairs(tbl.colspecs) do
    local width = count == 1 and 1 or (index == 1 and 0.35 or 0.65 / (count - 1))
    tbl.colspecs[index] = {spec[1], width}
  end
  return tbl
end
