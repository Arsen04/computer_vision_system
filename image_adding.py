from lib.Lib import Lib

functionality = Lib.functionality()
if functionality:
    try:
        vid = Lib.open_camera(functionality["db_path"])
    except Exception as e:
        print(e)
