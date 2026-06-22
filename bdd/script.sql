CREATE TABLE IF NOT EXISTS Cameras(
    id_camera    INTEGER PRIMARY KEY AUTOINCREMENT,
    nom          VARCHAR(50) NOT NULL,
    localisation VARCHAR(255)
);

INSERT OR IGNORE INTO Cameras (id_camera, nom, localisation) VALUES
    (1, 'Logitech C920', '12 rue des Colombes'),
    (2, 'OaK D Lite', '15 rue de la farandole');

CREATE TABLE IF NOT EXISTS Encombrants(
    id_encombrant    INTEGER PRIMARY KEY AUTOINCREMENT,
    classe           VARCHAR(50) NOT NULL,
    confiance        REAL NOT NULL,
    photo_path       VARCHAR(255) NOT NULL,
    zone_id          VARCHAR(50) NOT NULL,
    date_detection   DATETIME DEFAULT CURRENT_TIMESTAMP,
    statut           VARCHAR(50) NOT NULL DEFAULT 'present',
    date_suppression DATETIME,
    id_camera        INTEGER REFERENCES Cameras(id_camera)
);
