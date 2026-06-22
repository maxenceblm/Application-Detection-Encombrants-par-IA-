import sqlite3

DB_PATH = "bdd/encombrants.db"


def get_db_connection() :
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
 
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Cameras (
            id_camera    INTEGER PRIMARY KEY AUTOINCREMENT,
            nom          VARCHAR(50) NOT NULL,
            localisation VARCHAR(255),
            actif        INTEGER NOT NULL DEFAULT 0,
            latitude     REAL,
            longitude    REAL
        )
    """)
    conn.execute("""
        INSERT OR IGNORE INTO Cameras (id_camera, nom, localisation) VALUES
            (1, 'Caméra 1', 'Entrée principale'),
            (2, 'Caméra 2', 'Sortie arrière')
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Encombrants (
            id_encombrant    INTEGER PRIMARY KEY AUTOINCREMENT,
            classe           VARCHAR(50) NOT NULL,
            confiance        REAL NOT NULL,
            photo_path       VARCHAR(255) NOT NULL,
            zone_id          VARCHAR(50) NOT NULL,
            date_detection   DATETIME DEFAULT CURRENT_TIMESTAMP,
            statut           VARCHAR(50) NOT NULL DEFAULT 'Présent',
            date_suppression DATETIME,
            id_camera        INTEGER REFERENCES Cameras(id_camera)
        )
    """)
    conn.commit()
    conn.close()

def enregistrer_en_bdd(classe, confiance, photo_path, zone_id, id_camera):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO Encombrants (classe, confiance, photo_path, zone_id, id_camera) VALUES (?, ?, ?, ?, ?)",
        (classe, confiance, photo_path, zone_id, id_camera)
    )
    conn.commit()
    conn.close()

def get_encombrants_presents():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT e.*, c.nom AS camera_nom, c.localisation AS camera_localisation
        FROM Encombrants e
        LEFT JOIN Cameras c ON e.id_camera = c.id_camera
        WHERE e.statut = 'Présent'
        ORDER BY e.date_detection DESC
    """).fetchall()
    conn.close()
    return rows

def get_tous_encombrants():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT e.*, c.nom AS camera_nom, c.localisation AS camera_localisation
        FROM Encombrants e
        LEFT JOIN Cameras c ON e.id_camera = c.id_camera
        ORDER BY e.date_detection DESC
    """).fetchall()
    conn.close()
    return rows

def zone_deja_presente(zone_id):
    cx, cy = map(int, zone_id.split('_'))
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT zone_id FROM Encombrants WHERE statut = 'Présent'"
    ).fetchall()
    conn.close()
    for row in rows:
        zx, zy = map(int, row['zone_id'].split('_'))
        if abs(zx - cx) <= 1 and abs(zy - cy) <= 1:
            return True
    return False

def marquer_enleve(zone_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE Encombrants SET statut = 'Enlevé', date_suppression = CURRENT_TIMESTAMP WHERE zone_id = ? AND statut = 'Présent'",
        (zone_id,)
    )
    conn.commit()
    conn.close()

def reclasser_encombrant(id,classe) :
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE Encombrants SET classe = ? WHERE id_encombrant = ?", (classe,id,)
    )
    conn.commit()
    conn.close()

def supprimer_encombrant(id) :
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "DELETE FROM Encombrants WHERE id_encombrant = ?",
        (id,)
    )
    conn.commit()
    conn.close()

def set_camera_actif(id_camera, actif):
    conn = get_db_connection()
    conn.execute("UPDATE Cameras SET actif = ? WHERE id_camera = ?", (actif, id_camera))
    conn.commit()
    conn.close()

def get_encombrant_par_id(id_encombrant):
    conn = get_db_connection()
    row = conn.execute("""
        SELECT e.*, c.nom AS camera_nom, c.localisation AS camera_localisation
        FROM Encombrants e
        LEFT JOIN Cameras c ON e.id_camera = c.id_camera
        WHERE e.id_encombrant = ?
    """, (id_encombrant,)).fetchone()
    conn.close()
    return row

def marquer_enleve_par_id(id_encombrant):
    conn = get_db_connection()
    row = conn.execute("SELECT zone_id FROM Encombrants WHERE id_encombrant = ?", (id_encombrant,)).fetchone()
    if row:
        conn.execute(
            "UPDATE Encombrants SET statut = 'Enlevé', date_suppression = CURRENT_TIMESTAMP WHERE id_encombrant = ?",
            (id_encombrant,)
        )
        conn.commit()
    conn.close()

def get_cameras():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT c.*, COUNT(e.id_encombrant) AS nb_encombrants
        FROM Cameras c
        LEFT JOIN Encombrants e ON c.id_camera = e.id_camera AND e.statut = 'Présent'
        GROUP BY c.id_camera
        ORDER BY c.id_camera ASC
    """).fetchall()
    conn.close()
    return rows

def get_cameras_carte():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT c.id_camera, c.nom, c.localisation, c.latitude, c.longitude, c.actif,
               GROUP_CONCAT(DISTINCT e.id_encombrant || ':' || e.classe) AS classes
        FROM Cameras c
        LEFT JOIN Encombrants e ON c.id_camera = e.id_camera AND e.statut = 'Présent'
        WHERE c.latitude IS NOT NULL AND c.longitude IS NOT NULL
        GROUP BY c.id_camera
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]
