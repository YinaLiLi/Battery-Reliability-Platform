import subprocess
import time
from urllib.request import urlopen


def test_services():
    result = subprocess.run(['docker', 'compose', 'exec', '-T', 'kafka', '/opt/kafka/bin/kafka-topics.sh', '--bootstrap-server', 'kafka:29092', '--list'], capture_output=True, text=True, check=True)
    assert {'battery_measurements', 'battery_lifecycle'} <= set(result.stdout.split())
    subprocess.run(['docker', 'compose', 'exec', '-T', 'postgres', 'sh', '-ec', 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT count(*) FROM analytics.current_models"'], check=True, capture_output=True)
    for attempt in range(30):
        try:
            with urlopen('http://localhost:8501/_stcore/health', timeout=2) as response:
                assert response.status == 200
                return
        except OSError:
            time.sleep(2)
    raise AssertionError('Dashboard did not become healthy within 60 seconds')
