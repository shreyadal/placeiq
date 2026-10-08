# Interview experiences

One file per company. Inside a file, experiences are separated by a line containing
only `---`. Each experience starts with three header lines and then free text:

```
role: SDE Intern
year: 2025
concepts: DSA, OS
The first round was ...
---
```

`concepts` uses the same names as the question bank: DSA, OS, DBMS, CN, OOP, ML.

## These are samples, not real accounts

The files here were written by hand as **illustrative samples** so the retrieval and
generation code can be built and tested. They follow the widely known shape of campus
hiring at each kind of company, but they are not reports from real candidates and should
not be read as facts about any company's process.

Replace them with experiences you collect yourself (your seniors, your college placement
cell, your own interviews). If you take notes from public posts, write them up in your
own words rather than pasting the original text. After changing the files, rebuild the
index:

```
python -m rag.store
```
