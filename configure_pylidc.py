import configparser, os, pathlib
cfg = configparser.ConfigParser()
cfg['pylidc'] = {'dicom_path': str(pathlib.Path('data/raw/LIDC-IDRI').resolve())}
with open(os.path.expanduser('~/.pylidcrc'), 'w') as f:
    cfg.write(f)
print('pylidc configured')
