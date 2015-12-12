#!/bin/bash
function startEnv {
	$1 --clear $2
	$1 --system-site-packages $2
	source $2/bin/activate
	pip install pybrain
}
function installVirtualEnv {
	easy_install -d $1 virtualenv
}
if [[ ":$PYTHONPATH:" == *":$(pwd):"* ]]; then
	echo "Good PYTHONPATH!"
else
	export PYTHONPATH=$(pwd)
fi
if hash virtualenv 2>/dev/null; then
	startEnv virtualenv $(pwd)
else
	installVirtualEnv $(pwd)
	startEnv ./virtualenv $(pwd)
fi
