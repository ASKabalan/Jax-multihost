#!/bin/bash

# Default number of processes or first argument
num_processes=${1:-8}
shift # Move arguments to the left, so we can use the rest as test file filters

# Get the directory of the current script
script_dir=$(dirname "$(realpath "$0")")

# Find all test files in the script directory matching the pattern "test_*.py"
all_test_files=($(find "$script_dir" -name "test_*.py"))

# If additional arguments are provided, use them as filters for test files
if [ "$#" -gt 0 ]; then
  test_files=()
  for arg in "$@"; do
    for file in "${all_test_files[@]}"; do
      if [[ "$(basename "$file")" == "$arg" ]]; then
        test_files+=("$file")
      fi
    done
  done
else
  test_files=("${all_test_files[@]}")
fi

# Initialize arrays to store test names and outcomes
declare -a test_names
declare -a test_outcomes

# Function to run a test with MPI
run_mpi_test() {
  local test_file=$1  # The test file to run

  # Wrap the actual program call
  if [ "$OMPI_COMM_WORLD_RANK" = "0" ]; then
    python -m pytest "$test_file"
  else
    python -m pytest "$test_file" > /dev/null 2>&1
  fi
}

# Loop through all test files and run them with MPI
for test_file in "${test_files[@]}"; do
  # Use mpirun or mpiexec to launch the MPI job
  mpirun -np $num_processes bash -c "$(declare -f run_mpi_test); run_mpi_test '$test_file'"
  
  # Capture the exit status of mpirun/mpiexec
  exit_status=$?
  
  # Store test name and outcome
  test_names+=("$(basename "$test_file")")
  if [ $exit_status -ne 0 ]; then
    test_outcomes+=("failed")
  else
    test_outcomes+=("passed")
  fi
done

# Print test outcomes
echo "Test Outcomes:"
for i in "${!test_names[@]}"; do
  if [ "${test_outcomes[$i]}" = "failed" ]; then
    echo -e "\033[0;31mTest failed: ${test_names[$i]}\033[0m"
  else
    echo -e "\033[0;32mTest passed: ${test_names[$i]}\033[0m"
  fi
done

# Summary
echo "===================================================================================================="
failed_tests=$(printf '%s\n' "${test_outcomes[@]}" | grep -c "failed")
total_tests=${#test_names[@]}
if [ $failed_tests -eq 0 ]; then
    echo -e "\033[0;32mAll $total_tests tests successful\033[0m"
else
    echo -e "\033[0;31m$failed_tests/$total_tests tests failed\033[0m"
fi
