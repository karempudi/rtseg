

# Contains all the functions needed to 
# write to database of a live or simulated live
# experiment.

import sys
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_databases(save_dir, db_names):
    """
    Creates one database tables to store information at different stages of processing

    
    Arguments:
        save_dir: directory in which sqlite databases will be created, save_dir
        db_names: for each name in the list of db_names, a new database
            is created with exactly one table with the same name as the 
            database and we write with the schema designed only in this file
            and nowhere else. If you want to change the table schema, do it only
            in this file
    """
    for db in db_names:
        sys.stdout.write(f"Creating {db} datatable ... \n")
        sys.stdout.flush()
        db_file = save_dir / Path(db + '.db')
        # if the file exists don't create new table
        con = None
        try:
            if not db_file.exists(): 
                con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
            else:
                # find table if not create
                sys.stdout.write(f"{db} database exists. Will append to tables \n")
                sys.stdout.flush()
                con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)

            cur = con.cursor()
            if db == 'acquire_phase':
                cur.execute("""CREATE TABLE if not exists acquire_phase
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, position INTEGER, timepoint INTEGER)
                    """)
            elif db == 'acquire_fluor':
                cur.execute("""CREATE TABLE if not exists acquire_fluor
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, position INTEGER, timepoint INTEGER)
                    """)
            elif db == 'segment':
                cur.execute("""CREATE TABLE if not exists segment
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, position INTEGER, timepoint INTEGER,
                    rawpath_phase TEXT, rawpath_fluor TEXT, barcodes INTEGER, barcodelocations TEXT, numtraps INTEGER, traplocations TEXT)""")
            elif db == 'dots':
                cur.execute("""CREATE TABLE if not exists dots
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, position INTEGER, timepoint INTEGER,
                    channelno INTEGER, beforebarcode INTEGER, afterbarcode INTEGER, location INTEGER)""")
        except Exception as e:
            sys.stderr.write(f"Exception {e} raised in database creation :( \n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()


def write_to_db(event_data, dir_name, event_type):
    if event_type == 'acquire_phase':
        keys = ['position', 'timepoint']
        for key in keys:
            if key not in event_data:
                event_data[key] = None
        con = None
        db_file = dir_name / Path('acquire_phase.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None) 
            cur = con.cursor()
            cur.execute("""INSERT into acquire_phase (time, position, timepoint) VALUES (?, ?, ?)""", 
                        (datetime.now(), event_data['position'], event_data['timepoint'],))
        except Exception as e:
            sys.stderr.write(f"Error {e} while writing to table {event_type} -- {event_data}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
    elif event_type == 'acquire_fluor':
        keys = ['position', 'timepoint']
        for key in keys:
            if key not in event_data:
                event_data[key] = None
        con = None
        db_file = dir_name / Path('acquire_fluor.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None) 
            cur = con.cursor()
            cur.execute("""INSERT into acquire_fluor (time, position, timepoint) VALUES (?, ?, ?)""", 
                        (datetime.now(), event_data['position'], event_data['timepoint'],))
        except Exception as e:
            sys.stderr.write(f"Error {e} while writing to table {event_type} -- {event_data}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
        
    elif event_type == 'segment':
        keys = ['position', 'timepoint', 'num_traps', 'error']
        for key in keys:
            if key not in event_data:
                raise ValueError("Segmentation Database write received an invalid input ...")
        
        event_data['rawpath_phase'] = dir_name / Path('Pos'+ str(event_data['position'])) / Path('phase') /Path('phase_' + str(event_data['timepoint']).zfill(4)+ '.tiff')
        event_data['rawpath_fluor'] = dir_name / Path('Pos' + str(event_data['position']))  / Path('fluor') / Path('fluor_' + str(event_data['timepoint']).zfill(4) + '.tiff')
        #
    
        if event_data['error']:
            event_data['barcodes'] = 0
            event_data['barcode_locations'] = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
            event_data['trap_locations_list'] = [-1.]

        #print(json.dumps(event_data['barcode_locations']))
        #print(str(event_data['rawpath']))
        #print(type(event_data['barcode_locations']))
        #print(json.dumps(event_data['barcode_locations']))
        
        con = None
        db_file = dir_name / Path('segment.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None) 
            cur = con.cursor()
            cur.execute("""INSERT into segment (time, position, timepoint, rawpath_phase, rawpath_fluor, barcodes,
                        barcodelocations, numtraps, traplocations) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                        (datetime.now(), event_data['position'], event_data['timepoint'],
                        str(event_data['rawpath_phase']), str(event_data['rawpath_fluor']), 
                        len(event_data['barcode_locations']), json.dumps(event_data['barcode_locations'], cls=NpEncoder), 
                        int(event_data['num_traps']), json.dumps(event_data['trap_locations_list']))
                    )
        except Exception as e:
            sys.stderr.write(f"Error {e} while writing to table {event_type} -- {event_data}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
 
    elif event_type == 'dots':
        keys = ['position', 'timepoint']
        for key in keys:
            if key not in event_data:
                event_data[key] = None
        con = None
        db_file = dir_name / Path('track.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None) 
            cur = con.cursor()
            cur.execute("""INSERT into track (time, position, timepoint) VALUES (?, ?, ?)""", 
                        (datetime.now(), event_data['position'], event_data['time'])
                    )
        except Exception as e:
            sys.stderr.write(f"Error {e} while writing to table {event_type} -- {event_data}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
 

def read_from_db(event_type, dir_name, position=None, timepoint=None):
    if event_type == 'acquire':
        con = None
        position, timepoint = None, None
        db_file = dir_name / Path('acquire.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
            cur = con.cursor()
            cur.execute("SELECT position, timepoint FROM acquire ORDER BY id DESC LIMIT 1")

            position, timepoint = cur.fetchone()
        except Exception as e:
            sys.stdout.write(f"Error {e} while fetching from table {event_type} -- {dir_name}\n")
            sys.stdout.flush()

        finally:
            if con:
                con.close()
            return position, timepoint
            
    elif event_type == 'segment':
        con = None
        position, timepoint = None, None
        db_file = dir_name / Path('segment.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
            cur = con.cursor()
            cur.execute("SELECT position, timepoint FROM segment ORDER BY id DESC LIMIT 1")

            position, timepoint = cur.fetchone()

        except Exception as e:
            sys.stdout.write(f"Error {e} while fetching from table {event_type} -- {dir_name}\n")
            sys.stdout.flush()

        finally:
            if con:
                con.close()
            return position, timepoint

    elif event_type == 'track':
        con = None
        position, timepoint = None, None
        db_file = dir_name / Path('track.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
            cur = con.cursor()
            cur.execute("SELECT position, timepoint FROM track ORDER BY id DESC LIMIT 1")

            position, timepoint = cur.fetchone()

        except Exception as e:
            sys.stdout.write(f"Error {e} while fetching from table {event_type} -- {dir_name}\n")
            sys.stdout.flush()

        finally:
            if con:
                con.close()
            return position, timepoint
    elif event_type == 'barcode_locations':
        con = None
        # position and timepoint are the key work args
        db_file = dir_name / Path('segment.db')
        data = None
        #print(f"Getting barcode locations form {db_file}, position: {position}, timepoint: {timepoint}")
        try:
            con = sqlite3.connect(db_file)
            cur = con.cursor()
            cur.execute("""SELECT barcodes, barcodelocations, numchannels, channellocations FROM segment WHERE (position=? AND timepoint=?)""", (position, timepoint))
            data = cur.fetchone()
            data = {'numbarcodes': int(data[0]),
                    'barcode_locations': json.loads(data[1]),
                    'numchannels': int(data[2]),
                    'channel_locations': json.loads(data[3])}   
        except Exception as e:
            sys.stdout.write(f"Error {e} while fetching from table segment: barcode_locations -- {dir_name}\n")
            sys.stdout.flush()
        finally:
            if con:
                con.close()
            return data


def create_forkplots_db(save_dir, position):
    pass

def write_forkplots_db():
    pass

def read_forkplots_db():
    pass