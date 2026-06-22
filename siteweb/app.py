import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask , render_template , redirect , url_for, request , session, send_from_directory,jsonify
import sqlite3
from database import get_encombrants_presents , get_encombrant_par_id ,marquer_enleve_par_id, get_cameras , supprimer_encombrant , reclasser_encombrant , get_tous_encombrants,get_cameras_carte
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'Roxy13920'

@app.route('/')# Route de la page d'accueil
def index() :
    encombrants = get_encombrants_presents()
    return render_template('index.html', encombrants = encombrants)

@app.route('/encombrant/<int:id>')
def encombrant(id) :
    encombrant = get_encombrant_par_id(id)
    return render_template("encombrant.html",encombrant=encombrant)

@app.route('/marquer_enleve/<int:id>',methods=['POST'])
def enlever(id) :
    marquer_enleve_par_id(id)
    return redirect(url_for('index'))

@app.route('/supprimer/<int:id>',methods=['POST'])
def supprimer(id) :
    supprimer_encombrant(id)
    return redirect(url_for('index'))

@app.route('/cameras')
def cameras() :
    cameras = get_cameras()
    return render_template("cameras.html",cameras=cameras)

@app.route('/api/encombrants')
def api_encombrants():
    encombrants = get_encombrants_presents()
    return jsonify([dict(e) for e in encombrants])

@app.route('/reclasser/<int:id>',methods=['POST'])
def reclasser(id) : 
    classe = request.form['classe']
    reclasser_encombrant(id,classe)
    return redirect(url_for('encombrant',id=id))

@app.route('/historique')
def historique() :
    encombrants = get_tous_encombrants()
    return render_template('historique.html',encombrants=encombrants)

@app.template_filter('date_fr')
def date_fr(date_str):
    if not date_str:
        return '—'
    mois = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin',
            'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
    dt = datetime.strptime(date_str[:19], '%Y-%m-%d %H:%M:%S')
    return f"{dt.day} {mois[dt.month-1]} {dt.year} à {dt.hour:02d}h{dt.minute:02d}"

@app.route('/modele')
def modele():
    return render_template('modele.html')

@app.route('/api/cameras')
def api_cameras():
    return jsonify(get_cameras_carte())

@app.route('/carte')
def carte():
    return render_template('carte.html')

if __name__ == '__main__' :
    app.run(debug=True)
