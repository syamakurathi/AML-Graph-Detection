"""Execute and persist the notebook, streaming kernel progress to the console."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
path = root / 'HMPNN' / 'HMPNN_IBM_HI_Small.ipynb'
nb = nbformat.read(path, as_version=4)


class ProgressClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        result = super().process_message(msg, cell, cell_index)
        nbformat.write(self.nb, path)
        return result


km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = ProgressClient(nb, km=km, timeout=None, resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print('Executed notebook saved:', path)
