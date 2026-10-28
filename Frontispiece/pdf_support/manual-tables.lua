-- GFM tables otherwise use natural-width LaTeX columns, which do not wrap.
function Table(tbl)
  local count = #tbl.colspecs
  if count == 0 then return tbl end
  for index, spec in ipairs(tbl.colspecs) do
    local width = 1
    if count > 1 then
      width = index == 1 and 0.35 or 0.65 / (count - 1)
    end
    tbl.colspecs[index] = {spec[1], width}
  end
  return tbl
end
