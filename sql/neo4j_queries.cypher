// Q1. Simple: players with ELO above 2400
// "A basic node match with one property filter — the Cypher equivalent of a simple WHERE clause."
MATCH (p:Player)
WHERE p.elo > 2400
RETURN p.username, p.elo
ORDER BY p.elo DESC;


// Q2. Multi-condition: players who have won a Final AND have ELO above 2300
// "Traverses two relationships and applies two filters at once — only winners of Final matches who also rank above 2300 ELO."
MATCH (p:Player)-[r:PLAYED_IN]->(m:Match)-[:IN_TOURNAMENT]->(t:Tournament)
WHERE r.won = true AND m.round = 'Final' AND p.elo > 2300
RETURN p.username, p.elo, t.name AS tournament
ORDER BY p.elo DESC;


// Q3. Aggregate: total matches per player
// "Counts PLAYED_IN relationships per Player node — Cypher's equivalent of GROUP BY + COUNT."
MATCH (p:Player)-[:PLAYED_IN]->(m:Match)
RETURN p.username, COUNT(m) AS matches_played
ORDER BY matches_played DESC;
