#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

LOG_FILE="project_execution.log"

# Function to log and run commands
run_step() {
    local step_name="$1"
    local command="$2"
    
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "Starting Step: $step_name" | tee -a "$LOG_FILE"
    echo "Command: $command" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"
    echo "--------------------------------------------------------------------------------" | tee -a "$LOG_FILE"
    
    START_TIME=$(date +%s)
    
    # Run the command and append both stdout and stderr to the log
    if eval "$command" 2>&1 | tee -a "$LOG_FILE"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "--------------------------------------------------------------------------------" | tee -a "$LOG_FILE"
        echo "Step '$step_name' COMPLETED successfully." | tee -a "$LOG_FILE"
        echo "Duration: $DURATION seconds" | tee -a "$LOG_FILE"
        echo "================================================================================" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    else
        echo "--------------------------------------------------------------------------------" | tee -a "$LOG_FILE"
        echo "Step '$step_name' FAILED." | tee -a "$LOG_FILE"
        echo "Time: $(date)" | tee -a "$LOG_FILE"
        echo "================================================================================" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# Create/Clear log file
echo "Project Execution Log - Started at $(date)" > "$LOG_FILE"

# 0. Install Requirements
run_step "Install Dependencies (pip)" "pip install -r requirements.txt"

# 1. Download Data
run_step "Download PubMedQA Data" "python scripts/download_pubmedqa.py"

# 2. Prepare Corpus
run_step "Prepare PubMed Corpus" "python scripts/prepare_pubmed_corpus.py"

# 3. Fix Corpus IDs (Optional but recommended in README)
run_step "Fix Corpus IDs" "python scripts/fix_corpus_ids.py"

# 4. Create Indexes
run_step "Create DPR Index (This may take a while)" "python scripts/index.py"
run_step "Create SBERT Index (This may take a while)" "python scripts/build_sbert_index.py"

# 5. Run Experiments
run_step "Run LLM and RAG Experiments" "python scripts/run_experiments.py"

# 6. Evaluate
run_step "Evaluate F1 and Accuracy" "python scripts/evaluate_f1.py"
run_step "Evaluate Faithfulness" "python scripts/evaluate_faithfulness.py"

# 7. Analysis & Plots
run_step "Generate Analysis Plots" "python scripts/analysis_plots.py"

echo "All steps completed successfully! Check $LOG_FILE for full details."
