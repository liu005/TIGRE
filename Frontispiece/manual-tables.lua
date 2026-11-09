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

-- Permit long inline code and paths to wrap at their existing separators.
function Code(code)
  if FORMAT ~= 'latex' or #code.text <= 20 then return nil end
  local parts = {}
  local start = 1
  for index in code.text:gmatch('()[%._/\\%-]') do
    table.insert(parts, pandoc.Code(code.text:sub(start, index), code.attr))
    table.insert(parts, pandoc.RawInline('latex', '\\allowbreak{}'))
    start = index + 1
  end
  if #parts == 0 then return nil end
  if start <= #code.text then
    table.insert(parts, pandoc.Code(code.text:sub(start), code.attr))
  end
  return parts
end
