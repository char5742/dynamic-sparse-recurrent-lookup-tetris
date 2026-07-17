import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = path.resolve(import.meta.dirname, "..");
const inputPath = path.join(root, "1313", "NEXT性能変化.xlsx");
const outputDir = path.join(root, "artifacts", "1313_workbook");
await fs.mkdir(outputDir, { recursive: true });

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const overview = await workbook.inspect({
  kind: "workbook,sheet,table,drawing,definedName",
  maxChars: 12000,
  tableMaxRows: 12,
  tableMaxCols: 12,
  tableMaxCellChars: 120,
});
console.log("=== OVERVIEW ===");
console.log(overview.ndjson);

const sheetsInspection = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 6000,
});
const sheetRecords = sheetsInspection.ndjson
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line))
  .filter((record) => record.name);

for (const record of sheetRecords) {
  const sheetName = record.name;
  console.log(`=== SHEET ${sheetName} ===`);
  const region = await workbook.inspect({
    kind: "region,formula,drawing",
    sheetId: sheetName,
    range: "A1:AK30",
    maxChars: 7000,
    tableMaxRows: 30,
    tableMaxCols: 37,
    options: { maxResults: 300 },
  });
  console.log(region.ndjson);
  const rendered = await workbook.render({
    sheetName,
    range: "A1:AK30",
    scale: 1,
    format: "png",
  });
  const safeName = sheetName.replace(/[\\/:*?"<>|]/g, "_");
  await fs.writeFile(
    path.join(outputDir, `${safeName}.png`),
    new Uint8Array(await rendered.arrayBuffer()),
  );
}
