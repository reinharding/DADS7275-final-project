# MCSR Ranked — SQL & Graph Modeling

*English | [日本語](#日本語)*

Relational (MySQL) and graph (Neo4j) modeling of the Minecraft Speedrunning
Ranked playoff dataset, built on the same raw data used by the parent ML
project. Completed for **DADS 6700** at Northeastern.

The same MCSR data is modeled in **two paradigms**:

- **MySQL (relational):** 9 normalized tables with FK constraints, populated
  from scraped JSON/CSV by an ETL script. Best for analytical aggregation
  (per-season averages, win/loss tallies, top-N rankings).
- **Neo4j (graph):** the playoff brackets reshaped as
  `(:Player)-[:PLAYED_IN]->(:Match)-[:IN_TOURNAMENT]->(:Tournament)`. Best
  for relationship traversal (who has beaten whom, multi-hop paths through
  the bracket).

## MySQL component

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

[demo_queries.sql](demo_queries.sql) contains 10 worked queries covering the
core SQL patterns: simple SELECT, GROUP BY aggregation, INNER / LEFT JOIN,
nested subqueries, correlated subqueries, `>= ALL`, `EXISTS`, `UNION`,
subquery in `SELECT`, subquery in `FROM`.

## Neo4j component

[neo4j_setup.cypher](neo4j_setup.cypher) seeds a graph of 10 players, 8 seasons,
and 8 final matches, connected by `PLAYED_IN` and `IN_TOURNAMENT` edges.
[neo4j_queries.cypher](neo4j_queries.cypher) shows the Cypher equivalents of
filtering, multi-relationship traversal, and aggregation.

## Why both

Relational gives a clean answer to *"what is the average Elo per season?"* in
one `GROUP BY`. Graph gives a clean answer to *"which two players have faced
each other most often across all tournaments?"* without writing self-joins.
Modeling the same dataset twice is the point — each paradigm makes a
different question easy.

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
# PowerShell: $env:MYSQL_PASSWORD = 'your-password'

# 3. Load the data (reads from ../data/raw/ in the parent ML project)
pip install mysql-connector-python pandas
python load_mcsr_data.py

# 4. Run the demo dashboard
python app_dashboard.py
```

For Neo4j: paste [neo4j_setup.cypher](neo4j_setup.cypher) into Neo4j Browser to seed
the sandbox, then run queries from [neo4j_queries.cypher](neo4j_queries.cypher).

## Relation to the ML project

This SQL/Cypher work and the [parent ML project](../README.md) analyze the
same MCSR Ranked dataset from two angles: the ML side asks *"who will win
Season 10?"*, this side asks *"what does the bracket look like in normal
form, and what relationships does the graph reveal?"* The ETL deliberately
reads from `../data/raw/` so the database view is always in sync with the
data the ML notebook trains on.

---

# 日本語

*[English](#mcsr-ranked--sql--graph-modeling) | 日本語*

Minecraft Speedrunning Ranked プレイオフデータを、リレーショナル (MySQL) と
グラフ (Neo4j) の **2 つのパラダイム** でモデル化したプロジェクト。親 ML
プロジェクトと同じ生データを使用。Northeastern 大学 **DADS 6700** の課題として作成。

同じ MCSR データを次の 2 つの形でモデリング：

- **MySQL（リレーショナル）:** FK 制約付きの 9 テーブル正規化スキーマ。
  スクレイピング済み JSON/CSV から ETL スクリプトで投入。集計分析
  （シーズン別平均、勝敗集計、トップ N ランキング）に強い。
- **Neo4j（グラフ）:** プレイオフブラケットを
  `(:Player)-[:PLAYED_IN]->(:Match)-[:IN_TOURNAMENT]->(:Tournament)` の
  グラフ構造で表現。関係性の探索（誰が誰に勝ったか、ブラケットを跨いだマルチホップ経路）
  に強い。

## MySQL 部分

[mcsr_ddl.sql](mcsr_ddl.sql) で 9 テーブルを定義：

| テーブル | 目的 |
|---|---|
| `Player` | プレイヤー 1 人 1 行（UUID、ユーザー名、現在の Elo、自己ベスト） |
| `SeasonRanking` | シーズン別の Elo 履歴 |
| `PlayerStats` | 集計済み勝敗 / 平均完了時間 |
| `Tournament` / `Round` | シーズン毎のプレイオフブラケットとラウンド |
| `` `Match` `` | ラダー試合とトーナメント試合の両方 |
| `LadderMatch` / `TournamentMatch` | 試合をコンテキストに紐付けるサブタイプテーブル |
| `Participation` | 多対多：どの試合にどのプレイヤーが出場したか |
| `Split` | マイルストーン別タイム（ネザー入場、要塞、エンド、ドラゴン討伐） |

[demo_queries.sql](demo_queries.sql) には主要 SQL パターンを網羅した 10 種類のクエリ：
単純 SELECT、GROUP BY 集約、INNER / LEFT JOIN、ネストサブクエリ、相関サブクエリ、
`>= ALL`、`EXISTS`、`UNION`、SELECT 句サブクエリ、FROM 句サブクエリ。

## Neo4j 部分

[neo4j_setup.cypher](neo4j_setup.cypher) は 10 選手・8 シーズン・8 決勝戦を
`PLAYED_IN` と `IN_TOURNAMENT` リレーションで結ぶグラフを初期化。
[neo4j_queries.cypher](neo4j_queries.cypher) は、フィルタ・複数リレーション横断・
集約の Cypher 例を示す。

## なぜ両方やるか

「シーズン別の平均 Elo は？」のような問いは、リレーショナルなら `GROUP BY`
一発で綺麗に答えられる。「全トーナメントを通じて最も対戦回数の多い 2 人は？」
のような問いは、グラフなら自己結合を書かずに綺麗に答えられる。
**同じデータを 2 通りでモデル化すること自体が学びの目的** — それぞれのパラダイムが
得意とする問いが違う。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| [mcsr_ddl.sql](mcsr_ddl.sql) | スキーマ（9 テーブル、FK、ユニークキー） |
| [demo_queries.sql](demo_queries.sql) | 10 種のサンプルクエリ（JOIN、GROUP BY、サブクエリ） |
| [load_mcsr_data.py](load_mcsr_data.py) | ETL：`../data/raw/` を読み込み 9 テーブル投入、集計再計算 |
| [app_dashboard.py](app_dashboard.py) | MySQL 接続→3 クエリ実行→3 チャート保存 |
| [neo4j_setup.cypher](neo4j_setup.cypher) | グラフモデル：10 選手・8 トーナメント・8 決勝戦 |
| [neo4j_queries.cypher](neo4j_queries.cypher) | Cypher クエリ例 |
| [charts/](charts/) | `app_dashboard.py` の出力チャート |

## 実行方法

Python 3.10+ と ローカル MySQL 8 が必要。

```bash
# 1. スキーマを作成
mysql -u root -p < mcsr_ddl.sql

# 2. 認証情報を設定（host/user/db も環境変数で上書き可能）
export MYSQL_PASSWORD='your-password'      # Linux/macOS
# PowerShell: $env:MYSQL_PASSWORD = 'your-password'

# 3. データ投入（親 ML プロジェクトの ../data/raw/ から読み込む）
pip install mysql-connector-python pandas
python load_mcsr_data.py

# 4. ダッシュボード実行
python app_dashboard.py
```

Neo4j：[neo4j_setup.cypher](neo4j_setup.cypher) を Neo4j Browser に貼り付けて
サンドボックスを初期化、[neo4j_queries.cypher](neo4j_queries.cypher) のクエリを実行。

## ML プロジェクトとの関係

この SQL/Cypher プロジェクトと [親 ML プロジェクト](../README.md) は、
同じ MCSR Ranked データを 2 つの角度から分析している。ML 側の問いは
「シーズン 10 の優勝者は誰か？」、この SQL 側の問いは「ブラケットを正規形で
表現するとどうなるか、そしてグラフはどんな関係性を見せてくれるか？」。
ETL は意図的に `../data/raw/` を読みに行く設計なので、DB ビューは ML ノートブックが
学習するデータと常に同期している。
