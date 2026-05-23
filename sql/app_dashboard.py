# Part IV demo: connect to MySQL, run 3 queries, make 3 charts.

import os
import sys
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt

# Set MYSQL_PASSWORD in your environment before running.
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
if not MYSQL_PASSWORD:
    sys.exit("Set the MYSQL_PASSWORD environment variable before running this script.")

conn = mysql.connector.connect(
    host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
    port=int(os.environ.get("MYSQL_PORT", "3306")),
    user=os.environ.get("MYSQL_USER", "root"),
    password=MYSQL_PASSWORD,
    database=os.environ.get("MYSQL_DATABASE", "mcsr_ranked_playoffs"),
)
print("connected")

# query 1: top 10 players by elo
q1 = "SELECT username, elo FROM Player ORDER BY elo DESC LIMIT 10;"
df1 = pd.read_sql(q1, conn)
print("\ntop 10 by elo:")
print(df1)

# query 2: how many finals each player has won
q2 = """
SELECT p.username, COUNT(r.round_id) AS finals_won
FROM Player p
LEFT JOIN Round r ON r.winner_id = p.user_id AND r.round_name = 'Final'
GROUP BY p.user_id, p.username
HAVING finals_won > 0
ORDER BY finals_won DESC;
"""
df2 = pd.read_sql(q2, conn)
print("\nfinals won:")
print(df2)

# query 3: average elo each season
q3 = """
SELECT season, ROUND(AVG(elo)) AS avg_elo, COUNT(*) AS players
FROM SeasonRanking
GROUP BY season
ORDER BY season;
"""
df3 = pd.read_sql(q3, conn)
print("\navg elo per season:")
print(df3)

conn.close()

# chart 1: bar chart of top 10 elo
plt.figure()
plt.barh(df1["username"], df1["elo"])
plt.title("Top 10 Players by ELO")
plt.xlabel("ELO")
plt.tight_layout()
plt.savefig("chart1_top_elo.png")
plt.show()

# chart 2: bar chart of finals won
plt.figure()
plt.bar(df2["username"], df2["finals_won"])
plt.title("Finals Won per Player")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("chart2_finals_won.png")
plt.show()

# chart 3: line chart of avg elo per season
plt.figure()
plt.plot(df3["season"], df3["avg_elo"], marker="o")
plt.title("Average ELO per Season")
plt.xlabel("Season")
plt.ylabel("Avg ELO")
plt.tight_layout()
plt.savefig("chart3_avg_elo.png")
plt.show()

print("\ndone, charts saved")
