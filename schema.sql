-- Skapa tabeller för idrottsanalysprogram
CREATE TABLE Teams (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE Players (
    id SERIAL NOT NULL,
    team INTEGER REFERENCES Team(id) ON DELETE CASCADE,
    name VARCHAR(100),
    rightHanded BOOLEAN, 
    PRIMARY KEY (id,team)
);

CREATE TABLE Shots ( --inlc video
    player_id INTEGER REFERENCES Player(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    video VARCHAR(255) NOT NULL, -- filväg eller filnamn
    PRIMARY KEY (player_id, timestamp)
);

CREATE TYPE SHOTTYPE AS ENUM ('jump', 'set_shot', 'other');

CREATE TABLE ProcessedShots (
    player_id INTEGER,
    timestamp TIMESTAMP,
    stype SHOTTYPE,
    sspeed INTEGER,
    sthought TEXT,
    jheight INTEGER,
    jlength INTEGER,
    jbtime INTEGER,
    jpos INTEGER,
    lpos INTEGER,
    jang INTEGER,
    jpower INTEGER,
    PRIMARY KEY (player_id, timestamp),
    FOREIGN KEY (player_id, timestamp) REFERENCES Shots(player_id, timestamp) ON DELETE CASCADE
);