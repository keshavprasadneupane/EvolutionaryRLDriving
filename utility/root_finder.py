from pathlib import Path

def find_project_root(markers=(".git", "pyproject.toml", "setup.py", "requirements.txt", "README.md")):
	"""
	Finds the project root by checking the script's location first, 
	then falling back to the current working directory.
	"""
	# Try using the script's location if __file__ is available
	try:
		current_dir = Path(__file__).resolve().parent
	except NameError:
		# Fallback for interactive shells / Jupyter notebooks
		current_dir = Path.cwd().resolve()
	
	# Walk upward through parents
	for parent in [current_dir] + list(current_dir.parents):
		if any((parent / marker).exists() for marker in markers):
			return parent
			
	return current_dir