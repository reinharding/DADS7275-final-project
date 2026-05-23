USE mcsr_ranked_playoffs;


-- Q1. Simple: top 10 players by ELO
-- "A basic SELECT from one table — pulling the ten highest-rated players in the ladder."
SELECT username, elo
FROM Player
ORDER BY elo DESC
LIMIT 10;


-- Q2. Aggregate: avg ELO and player count per season
-- "GROUP BY season and averaging — shows how the competitive level shifted across seasons."
SELECT season,
       COUNT(*) AS players,
       ROUND(AVG(elo)) AS avg_elo
FROM SeasonRanking
GROUP BY season
ORDER BY season;


-- Q3. Outer join: every player with their tournament-final wins (incl. 0)
-- "LEFT JOIN Player to Round so players with zero wins still show up."
SELECT p.username, COUNT(r.round_id) AS finals_won
FROM Player p
LEFT JOIN Round r ON r.winner_id = p.user_id AND r.round_name = 'Final'
GROUP BY p.user_id, p.username
ORDER BY finals_won DESC;


-- Q4. Inner join: winner of each tournament Final
-- "Inner join between Round and Player — only Finals that actually have a winner come through."
SELECT r.round_name, p.username AS winner
FROM Round r
JOIN Player p ON p.user_id = r.winner_id
WHERE r.round_name = 'Final';


-- Q5. Nested: players above the global average ELO
-- "The inner SELECT computes the league-wide average once, the outer query filters anyone above it."
SELECT username, elo
FROM Player
WHERE elo > (SELECT AVG(elo) FROM Player)
ORDER BY elo DESC;


-- Q6. Correlated: each player with their personal match count (only those who played)
-- "Both subqueries reference the outer p.user_id — MySQL recomputes the COUNT per row, giving each player their own participation total. This is the textbook correlated pattern."
SELECT p.username,
       p.elo,
       (SELECT COUNT(*)
        FROM Participation pa
        WHERE pa.user_id = p.user_id) AS matches_played
FROM Player p
WHERE (SELECT COUNT(*)
       FROM Participation pa
       WHERE pa.user_id = p.user_id) >= 1
ORDER BY matches_played DESC
LIMIT 20;


-- Q7a. >=ALL: the player(s) with the highest ELO
-- "ELO must be at least as high as every value the subquery returns — this pins the very top of the ladder."
SELECT username, elo
FROM Player
WHERE elo >= ALL (SELECT elo FROM Player);


-- Q7b. EXISTS: players who have won at least one Final
-- "EXISTS just tests whether the inner query returns anything — every past tournament champion shows up."
SELECT p.username
FROM Player p
WHERE NOT EXISTS (
    SELECT * FROM Round r
    WHERE r.winner_id = p.user_id AND r.round_name = 'Final'
)
LIMIT 10;


-- Q8. Union: players who appeared in a tournament match OR a ladder match
-- "UNION merges two result sets and removes duplicates — one combined roster across both match types."
SELECT pl.username
FROM Participation p
JOIN TournamentMatch tm ON tm.match_id = p.match_id
JOIN Player pl ON pl.user_id = p.user_id
UNION
SELECT pl.username
FROM Participation p
JOIN LadderMatch lm ON lm.match_id = p.match_id
JOIN Player pl ON pl.user_id = p.user_id;


-- Q9. Subquery in SELECT: inline win count per player
-- "That count column is its own subquery running per row — we get each player's ELO and total wins in one result."
SELECT p.username,
       p.elo,
       (SELECT COUNT(*) FROM `Match` m WHERE m.winner_id = p.user_id) AS wins
FROM Player p
ORDER BY wins DESC
LIMIT 10;


-- Q10. Subquery in FROM: seasons ranked by avg ELO
-- "The subquery builds a per-season summary — the outer query treats it as a table and ranks it."
SELECT season, avg_elo
FROM (
    SELECT season, ROUND(AVG(elo)) AS avg_elo
    FROM SeasonRanking
    GROUP BY season
) AS s
ORDER BY avg_elo DESC;
