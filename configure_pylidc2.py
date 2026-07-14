import configparser, os, pathlib
cfg = configparser.ConfigParser()
cfg['dicom'] = {'path': str(pathlib.Path('data/raw/LIDC-IDRI').resolve())}
conf_file = os.path.join(os.path.expanduser('~'), 'pylidc.conf')
with open(conf_file, 'w') as f:
    cfg.write(f)
print('pylidc configured at', conf_file)
