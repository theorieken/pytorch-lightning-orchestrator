import argparse
from deep_orchestrator.orchestrator import Orchestrator

# Create a parser
parser = argparse.ArgumentParser(description='Run orchestrator with job configs.')

# Add argument
parser.add_argument('configs', metavar='C', type=str, nargs='+', help='Job configuration file(s)')

# Parse arguments
args = parser.parse_args()

# Instantiate orchestrator and run
orchestrator = Orchestrator(jobs=args.configs)
orchestrator.run()
