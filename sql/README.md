# MCSR Ranked — SQL & Graph Modeling

Relational (MySQL) and graph (Neo4j) modeling of the Minecraft Speedrunning
Ranked playoff dataset, built on the same raw data used by the parent ML
project. Completed for **DADS 6700** at Northeastern.

The relational schema and ETL load the scraped JSON/CSV in
[`../data/raw/`](../data/raw/) into a normalized 9-table database; the
dashboard runs three example queries and renders three charts.

## Schema

Nine tables in [mcsr_ddl.sql](mcsr_ddl.sql):

| Table | Purpose |
|---|---|
| `Player` | One row per player (UUID, username, current Elo, personal best) |
| `SeasonRanking` | Per-season Elo for each player |
| `PlayerStats` | Aggregate W/L/D and average finish time |
| `Tournament` / `Round` | Playoff brackets per season and per round |
| `` `Match` `` | Both ladder games and tournament games |
| `LadderMatch` / `TournamentMatch` | Subtype tables linking matches to context |
| `Participation` | Many-to-many: which players appeared in which match |
| `Split` | Per-milestone times (nether entry, fortress, end, dragon) |

A graph-database alternative lives in
[neo4j_setup.cypher](neo4j_setup.cypher) and [neo4j_queries.cypher](neo4j_queries.cypher) —
the same playoff finals modeled as `(:Player)-[:PLAYED_IN]->(:Match)-[:IN_TOURNAMENT]->(:Tournament)`.

## Files

| File | What |
|---|---|
| [mcsr_ddl.sql](mcsr_ddl.sql) | Schema (9 tables, FKs, unique keys) |
| [demo_queries.sql](demo_queries.sql) | 10 example queries (joins, group-by, subqueries) |
| [load_mcsr_data.py](load_mcsr_data.py) | ETL: reads `../data/raw/`, populates all 9 tables, backfills aggregates |
| [app_dashboard.py](app_dashboard.py) | Connects to MySQL, runs 3 queries, saves 3 charts |
| [neo4j_setup.cypher](neo4j_setup.cypher) | Graph model: 10 players, 8 tournaments, 8 finals |
| [neo4j_queries.cypher](neo4j_queries.cypher) | Example Cypher queries |
| [charts/](charts/) | Output charts from `app_dashboard.py` |

## How to run

Requires Python 3.10+ and a local MySQL 8 instance.

```bash
# 1. Create the schema
mysql -u root -p < mcsr_ddl.sql

# 2. Set credentials (override host/user/db too if you need to)
export MYSQL_PASSWORD='your-password'      # Linux/macOS
# or PowerShell: $env:MYSQL_PASSWORD = 'your-password'

# 3. Load the data (reads from ../data/raw/ in the parent ML project)
pip install mysql-connector-python pandas
python load_mcsr_data.py

# 4. Run the demo dashboard
python app_dashboard.py
```

For Neo4j: paste [neo4j_setup.cypher](neo4j_setup.cypher) into Neo4j Browser to seed
the sandbox, then run queries from [neo4j_queries.cypher](neo4j_queries.cypher).

## Relation to the ML project

This SQL work and the [parent ML project](../README.md) analyze the same
MCSR Ranked dataset from two angles: the ML side asks "who will win
Season 10?", this side asks "what does the bracket look like in normal form?"
The ETL deliberately reads from `../data/raw/` so the relational view is
always in sync with the data the ML notebook trains on.
