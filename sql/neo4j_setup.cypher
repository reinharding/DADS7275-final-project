// Wipe the sandbox before loading
MATCH (n) DETACH DELETE n;

// 10 players + 8 tournaments + 8 final matches + relationships
CREATE
  (edcr:Player   {username: 'edcr',          elo: 2600}),
  (hn:Player     {username: 'hackingnoises', elo: 2480}),
  (fein:Player   {username: 'Feinberg',      elo: 2462}),
  (steez:Player  {username: 'steez',         elo: 2432}),
  (bing:Player   {username: 'bing_pigs',     elo: 2409}),
  (infume:Player {username: 'Infume',        elo: 2397}),
  (lowkey:Player {username: 'lowk3y_',       elo: 2387}),
  (dog:Player    {username: 'doogile',       elo: 2385}),
  (silver:Player {username: 'silverrruns',   elo: 2260}),
  (anco:Player   {username: 'Ancoboyy',      elo: 2179}),

  (t1:Tournament {name: 'Season 1', year: 2021}),
  (t2:Tournament {name: 'Season 2', year: 2022}),
  (t4:Tournament {name: 'Season 4', year: 2024}),
  (t5:Tournament {name: 'Season 5', year: 2025}),
  (t6:Tournament {name: 'Season 6', year: 2026}),
  (t7:Tournament {name: 'Season 7', year: 2027}),
  (t8:Tournament {name: 'Season 8', year: 2028}),
  (t9:Tournament {name: 'Season 9', year: 2029}),

  (m1:Match {match_id: 4546732, round: 'Final'}),
  (m2:Match {match_id: 4546718, round: 'Final'}),
  (m3:Match {match_id: 4546704, round: 'Final'}),
  (m4:Match {match_id: 4546693, round: 'Final'}),
  (m5:Match {match_id: 4546680, round: 'Final'}),
  (m6:Match {match_id: 4546667, round: 'Final'}),
  (m7:Match {match_id: 4546653, round: 'Final'}),
  (m8:Match {match_id: 4546638, round: 'Final'}),

  // Season 1 Final: silverrruns beat doogile
  (silver)-[:PLAYED_IN {won: true}]->(m1),
  (dog)-[:PLAYED_IN    {won: false}]->(m1),
  (m1)-[:IN_TOURNAMENT]->(t1),

  // Season 2 Final: lowk3y_ beat silverrruns
  (lowkey)-[:PLAYED_IN {won: true}]->(m2),
  (silver)-[:PLAYED_IN {won: false}]->(m2),
  (m2)-[:IN_TOURNAMENT]->(t2),

  // Season 4 Final: lowk3y_ beat Ancoboyy
  (lowkey)-[:PLAYED_IN {won: true}]->(m3),
  (anco)-[:PLAYED_IN   {won: false}]->(m3),
  (m3)-[:IN_TOURNAMENT]->(t4),

  // Season 5 Final: doogile beat hackingnoises
  (dog)-[:PLAYED_IN {won: true}]->(m4),
  (hn)-[:PLAYED_IN  {won: false}]->(m4),
  (m4)-[:IN_TOURNAMENT]->(t5),

  // Season 6 Final: lowk3y_ beat doogile
  (lowkey)-[:PLAYED_IN {won: true}]->(m5),
  (dog)-[:PLAYED_IN    {won: false}]->(m5),
  (m5)-[:IN_TOURNAMENT]->(t6),

  // Season 7 Final: doogile beat lowk3y_
  (dog)-[:PLAYED_IN    {won: true}]->(m6),
  (lowkey)-[:PLAYED_IN {won: false}]->(m6),
  (m6)-[:IN_TOURNAMENT]->(t7),

  // Season 8 Final: hackingnoises beat Infume
  (hn)-[:PLAYED_IN     {won: true}]->(m7),
  (infume)-[:PLAYED_IN {won: false}]->(m7),
  (m7)-[:IN_TOURNAMENT]->(t8),

  // Season 9 Final: hackingnoises beat doogile
  (hn)-[:PLAYED_IN  {won: true}]->(m8),
  (dog)-[:PLAYED_IN {won: false}]->(m8),
  (m8)-[:IN_TOURNAMENT]->(t9);
