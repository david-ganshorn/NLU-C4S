from __future__ import print_function
import requests

body = requests.post('https://sap.datascience.com/deploy/deploy-test-320484-v1/',
    json={"new_review":["we had a great day"]},
    cookies={
        'datascience-platform': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyODQ2ZTBiOC04Zjc0LTQ3YWItOGU0Zi00NDgzNTk2NTRjOGUiLCJzZXJ2aWNlTmFtZSI6ImRlcGxveS10ZXN0LTMyMDQ4NC12MSIsImlhdCI6MTUyMTAzMjY1N30.TEcFltf1aEpkQGrIZBR1P5KXoy4uf1eYqP8TprKmVis'
    },
)

print(body.text)
