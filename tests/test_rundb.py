import mlrun
from conftest import rundb_path, results


def get_db():
    return mlrun.get_run_db(rundb_path).connect()

#
#pprint.pprint(db.list_runs()[:2])

def test_list_runs():

    db = get_db()
    runs = db.list_runs()
    assert runs, 'empty runs result'

    html = runs.show(display=False)

    with open(f'{results}/runs.html', 'w') as fp:
        fp.write(html)


def test_list_artifacts():

    db = get_db()
    artifacts = db.list_artifacts()
    assert artifacts, 'empty artifacts result'

    html = artifacts.show(display=False)

    with open('{}/artifacts.html'.format(results), 'w') as fp:
        fp.write(html)




