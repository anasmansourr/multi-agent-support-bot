# Troubleshooting performance, uploads and imports

Slow pages: use a supported browser, clear the cache, disable browser extensions and check the status page. Boards with more than 1,000 tasks load faster when you apply a filter.

Uploads: the maximum file size is 25 MB. Executable files such as .exe and .bat are blocked. An "Upload failed" banner usually means the file is too large, the type is blocked or the connection dropped. For larger files, upload them to cloud storage and attach the link.

CSV import: the file must be UTF-8 with a header row and comma separators, with at most 10,000 rows. An error on row 1 usually means the header row is missing or the separator is not a comma.

CSV export with garbled characters: open the file in Excel with Data > From Text/CSV and choose UTF-8.
