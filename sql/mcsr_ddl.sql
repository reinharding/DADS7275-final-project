CREATE DATABASE IF NOT EXISTS mcsr_ranked_playoffs;
USE mcsr_ranked_playoffs;

CREATE TABLE Player (
    user_id CHAR(36) NOT NULL,
    username VARCHAR(50) NOT NULL,
    elo INT NOT NULL,
    personal_best INT,
    PRIMARY KEY (user_id),
    UNIQUE KEY (username)
); 

CREATE TABLE SeasonRanking (
    ranking_id INT NOT NULL AUTO_INCREMENT,
    user_id CHAR(36) NOT NULL,
    season INT NOT NULL,
    elo INT NOT NULL,
    PRIMARY KEY (ranking_id),
    UNIQUE KEY (user_id, season),
    FOREIGN KEY (user_id) REFERENCES Player(user_id) ON DELETE CASCADE
);

CREATE TABLE PlayerStats (
    stat_id INT NOT NULL AUTO_INCREMENT,
    user_id CHAR(36) NOT NULL,
    average_time INT,
    win_count INT NOT NULL DEFAULT 0,
    loss_count INT NOT NULL DEFAULT 0,
    draw_count INT NOT NULL DEFAULT 0,
    game_count INT NOT NULL DEFAULT 0,
    PRIMARY KEY (stat_id),
    UNIQUE KEY (user_id),
    FOREIGN KEY (user_id) REFERENCES Player(user_id)
);

CREATE TABLE Tournament (
    tournament_id INT NOT NULL AUTO_INCREMENT,
    tournament_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    prize_money DECIMAL(10, 2),
    PRIMARY KEY (tournament_id)
);

CREATE TABLE Round (
    round_id INT NOT NULL AUTO_INCREMENT,
    tournament_id INT NOT NULL,
    round_name VARCHAR(50) NOT NULL,
    winner_id CHAR(36),
    PRIMARY KEY (round_id),
    FOREIGN KEY (tournament_id) REFERENCES Tournament(tournament_id),
    FOREIGN KEY (winner_id) REFERENCES Player(user_id) ON DELETE SET NULL
);

CREATE TABLE `Match` (
    match_id INT NOT NULL AUTO_INCREMENT,
    world_seed BIGINT,
    match_date DATETIME NOT NULL,
    winner_id CHAR(36),
    PRIMARY KEY (match_id),
    FOREIGN KEY (winner_id) REFERENCES Player(user_id)
);

CREATE TABLE TournamentMatch (
    match_id INT NOT NULL,
    round_id INT NOT NULL,
    PRIMARY KEY (match_id),
    FOREIGN KEY (match_id) REFERENCES `Match`(match_id) ON DELETE CASCADE,
    FOREIGN KEY (round_id) REFERENCES Round(round_id)
);

CREATE TABLE LadderMatch (
    match_id INT NOT NULL,
    season INT NOT NULL,
    PRIMARY KEY (match_id),
    FOREIGN KEY (match_id) REFERENCES `Match`(match_id) ON DELETE CASCADE
);

CREATE TABLE Participation (
    user_id CHAR(36) NOT NULL,
    match_id INT NOT NULL,
    PRIMARY KEY (user_id, match_id),
    FOREIGN KEY (user_id) REFERENCES Player(user_id),
    FOREIGN KEY (match_id) REFERENCES `Match`(match_id) ON DELETE CASCADE
);

CREATE TABLE Split (
    split_id INT NOT NULL AUTO_INCREMENT,
    user_id CHAR(36) NOT NULL,
    match_id INT NOT NULL,
    split_name VARCHAR(50) NOT NULL,
    split_time INT NOT NULL,
    time_difference INT,
    PRIMARY KEY (split_id),
    FOREIGN KEY (user_id, match_id) REFERENCES Participation(user_id, match_id) ON DELETE CASCADE
);
