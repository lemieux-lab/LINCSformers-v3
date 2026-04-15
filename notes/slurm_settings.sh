# for reference

# >>> slurm helper setup >>>
_slurm_submit () {
    local job_name="$1"
    local cpus="$2"
    local gpu="$3"
    local mem="$4"
    shift 4

    if [ $# -lt 1 ]; then
        echo -e "\e[1;31mError: Missing arguments.\e[0m"
        echo "Usage: _slurm_submit JOB_NAME CPUS GPU MEM COMMAND..."
        return 1
    fi

    local time_limit="${SLURM_TIME:-}"
    local partition="${SLURM_PARTITION:-}"

    local timestamp
    local log_dir="/home/golem/scratch/chans/lincsv4/slurm/logs"
    local log_path
    local cmd=("$@")

    mkdir -p "$log_dir"
    timestamp=$(date +"%Y%m%d_%H%M")
    log_path="${log_dir}/${job_name}_${timestamp}.out"

    printf "\n"
    printf "\e[1;36m========================================\e[0m\n"
    printf "\e[1;32mSubmitting SLURM job:\e[0m\n"
    printf "\e[1;36m========================================\e[0m\n"
    printf "\e[1;33mName       :\e[0m %s\n" "$job_name"
    printf "\e[1;33mCPUs       :\e[0m %s\n" "$cpus"
    printf "\e[1;33mGPU        :\e[0m %s\n" "$gpu"
    printf "\e[1;33mMemory     :\e[0m %s\n" "$mem"
    [ -n "$time_limit" ] && printf "\e[1;33mTime       :\e[0m %s\n" "$time_limit"
    [ -n "$partition" ]  && printf "\e[1;33mPartition  :\e[0m %s\n" "$partition"
    printf "\e[1;33mLogs       :\e[0m %s\n" "$log_path"
    printf "\e[1;33mCommand    :\e[0m %s\n" "${cmd[*]}"
    printf "\e[1;36m========================================\e[0m\n\n"

    local sbatch_args=(
        --cpus-per-task="$cpus"
        --gres="$gpu"
        --mem="$mem"
        --job-name="$job_name"
        --output="$log_path"
    )

    if [[ -n "$time_limit" && "$time_limit" != "0" ]]; then
        sbatch_args+=(--time="$time_limit")
    fi

    if [ -n "$partition" ]; then
        sbatch_args+=(--partition="$partition")
    fi

    sbatch "${sbatch_args[@]}" --wrap "${cmd[*]}"
}

tera_sbatch()  { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}" 100 gpu:1         1000G "${@:2}"; }
giga_sbatch()  { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}"  50 gpu:1          500G "${@:2}"; }
mega_sbatch()  { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}"  20 gpu:1          100G "${@:2}"; }
kilo_sbatch()  { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}"  10 gpu:1           50G "${@:2}"; }
micro_sbatch() { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}"   2 gpu:1           25G "${@:2}"; }

v100_sbatch()  { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}"  16 gpu:V100:1      90G "${@:2}"; }
l40b_sbatch()  { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}" 128 gpu:L40S:1     800G "${@:2}"; }
l40m_sbatch()  { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}"  64 gpu:L40S:1     400G "${@:2}"; }
l40s_sbatch()  { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}"  32 gpu:L40S:1     200G "${@:2}"; }
gh200_sbatch() { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}"  64 gpu:GH200:1    500G "${@:2}"; }
gh18_sbatch()  { _slurm_submit "${1:-$(date +%Y%m%d_%H%M)}"  16 gpu:1g.18gb:1  125G "${@:2}"; }

var_sbatch() {
    if [ $# -lt 4 ]; then
        echo -e "\e[1;31mError: Missing arguments.\e[0m"
        echo "Usage: var_sbatch JOB_NAME CPUS GPU MEM COMMAND..."
        return 1
    fi
    local job_name="$1"
    local cpus="$2"
    local gpu="$3"
    local mem="$4"
    shift 4
    
    _slurm_submit "$job_name" "$cpus" "$gpu" "$mem" "$@"
}
slurm_presets() {
    printf "\n"
    printf "\e[1;36m========================================================================================\e[0m\n"
    printf "\e[1;32mAvailable SLURM presets\e[0m\n"
    printf "\e[1;36m========================================================================================\e[0m\n"

    printf "\e[1;33m%-15s %-10s %-15s %-10s\e[0m\n" "Command" "CPUs" "GPU" "Memory"
    printf "%-15s %-10s %-15s %-10s\n" "--------------" "----------" "---------------" "----------"

    printf "%-15s %-10s %-15s %-10s\n" "micro_sbatch" "2"   "gpu:1"         "25G"
    printf "%-15s %-10s %-15s %-10s\n" "kilo_sbatch"  "10"  "gpu:1"         "50G"
    printf "%-15s %-10s %-15s %-10s\n" "mega_sbatch"  "20"  "gpu:1"         "100G"
    printf "%-15s %-10s %-15s %-10s\n" "giga_sbatch"  "50"  "gpu:1"         "500G"
    printf "%-15s %-10s %-15s %-10s\n" "tera_sbatch"  "100" "gpu:1"         "1000G"
    printf "%-15s %-10s %-15s %-10s\n" "v100_sbatch"  "16"  "gpu:V100:1"    "90G"
    printf "%-15s %-10s %-15s %-10s\n" "l40b_sbatch"  "128" "gpu:L40S:1"    "800G"
    printf "%-15s %-10s %-15s %-10s\n" "l40m_sbatch"  "64"  "gpu:L40S:1"    "400G"
    printf "%-15s %-10s %-15s %-10s\n" "l40s_sbatch"  "32"  "gpu:L40S:1"    "200G"
    printf "%-15s %-10s %-15s %-10s\n" "gh200_sbatch" "64"  "gpu:GH200:1"   "500G"
    printf "%-15s %-10s %-15s %-10s\n" "gh18_sbatch"  "16"  "gpu:1g.18gb:1" "125G"

    printf "\n"
    printf "\e[1;33mUsage:\e[0m  [PRESET] [JOB_NAME] [COMMAND]\n"
    printf "\e[1;33mCustom:\e[0m var_sbatch [JOB_NAME] [CPUS] [GPU] [MEM] [COMMAND]...\n"
    printf "\e[1;36m========================================================================================\e[0m\n\n"
}

slog() {
    local name="${1:-}"
    local file
    local log_dir="/home/golem/scratch/chans/lincsv4/slurm/logs"

    if [ -z "$name" ]; then
        file=$(command ls -t "$log_dir"/*.out 2>/dev/null | head -n 1)
        if [ -z "$file" ]; then
            echo "No log files found."
            return 1
        fi
        echo "No job name provided, using newest log file."
    else
        file=$(command ls -t "$log_dir"/"${name}"_*.out 2>/dev/null | head -n 1)
        if [ -z "$file" ]; then
            echo "No log found for job name: $name"
            return 1
        fi
    fi

    echo "Showing: $file"
    cat "$file"
}

sclear() {
    local log_dir="/home/golem/scratch/chans/lincsv4/slurm/logs"
    local file base name
    local -A active_jobs

    while read -r name; do
        [ -n "$name" ] && active_jobs["$name"]=1
    done < <(squeue -u "$USER" -h -o "%j")

    shopt -s nullglob
    for file in "$log_dir"/*.out "$log_dir"/*.err; do
        base=$(basename "$file")

        name="${base%.*}"
        name="${name%_*}"
        name="${name%_*}"

        if [ -z "${active_jobs[$name]:-}" ]; then
            echo "Removing: $file"
            rm -f -- "$file"
        fi
    done
    shopt -u nullglob
}

sarchive() {
    local log_dir="/home/golem/scratch/chans/lincsv4/slurm/logs"
    local archive_dir="${log_dir}/archive"
    local file base name
    local -A active_jobs

    mkdir -p "$archive_dir"

    while read -r name; do
        [ -n "$name" ] && active_jobs["$name"]=1
    done < <(squeue -u "$USER" -h -o "%j")

    shopt -s nullglob
    for file in "$log_dir"/*.out "$log_dir"/*.err; do
        base=$(basename "$file")

        name="${base%.*}"
        name="${name%_*}"
        name="${name%_*}"

        if [ -z "${active_jobs[$name]:-}" ]; then
            echo "Archiving: $file"
            mv -- "$file" "$archive_dir/"
        fi
    done
    shopt -u nullglob
}

alias sp="slurm_presets"
alias spresets="slurm_presets"
alias sq="squeue -u chans"
# <<< slurm helper setup <<<

# >>> training runs >>>
run_finetune() {
    local target_preset="$1"
    local mode="$2"
    local level="$3"
    local epochs="$4"
    local model="$5"
    local dataset="$6"
    local note="$7"
    local opt_time="$8"

    if [ $# -lt 7 ]; then
        echo -e "\e[1;31mError: Missing arguments.\e[0m"
        echo "Usage: run_finetune [PRESET] [MODE] [LEVEL] [EPOCHS] [MODEL] [DATASET] [NOTE] [OPTIONAL_TIME]"
        return 1
    fi

    if ! type "$target_preset" &> /dev/null; then
        echo -e "\e[1;31mPreset '$target_preset' does not exist.\e[0m"
        return 1
    fi

    local julia_script=""
    local SLURM_TIME=""
    
    if [[ "$mode" == "emb" ]]; then
        julia_script="scripts/finetune/main_emb.jl"
        SLURM_TIME="1-00:00:00"
    elif [[ "$mode" == "e2e" ]]; then
        SLURM_TIME="4-00:00:00"
        if [[ "$model" == "mlp" ]]; then
            julia_script="scripts/finetune/main_mlp.jl"
        else
            julia_script="scripts/finetune/main_tf.jl"
        fi
    else
        echo -e "\e[1;31mError: mode must be 'emb' or 'e2e'\e[0m"
        return 1
    fi

    if [ -n "$opt_time" ]; then
        SLURM_TIME="$opt_time"
    fi

    local job_name="${mode^^}_${level}_${epochs}_${model}_${dataset}"

    local cmd="

ARCH=\$(uname -m)
if [[ \"\$ARCH\" == \"aarch64\" || \"\$ARCH\" == \"arm64\" ]]; then
    ENV_PATH=\"/home/golem/scratch/chans/lincsv3/aarch64\"
else
    ENV_PATH=\"/home/golem/scratch/chans/lincsv3\"
fi

echo \"architecture: \$ARCH\"
echo \"environment: \$ENV_PATH\"

cd /home/golem/scratch/chans/lincsv3
julia --project=\"\$ENV_PATH\" -e 'using Pkg; Pkg.instantiate()'

julia --project=\"\$ENV_PATH\" \"$julia_script\" \
    --modeltype \"$model\" \
    --dataset \"$dataset\" \
    --level \"$level\" \
    --n_epochs \"$epochs\" \
    --additional_notes \"$note\"
"

    $target_preset "$job_name" "$cmd"
}

run_pretrain() {
    local target_preset="$1"
    local epochs="$2"
    local model="$3"
    local dataset="$4"
    local note="$5"
    local opt_time="$6"

    if [ $# -lt 5 ]; then
        echo -e "\e[1;31mError: Missing arguments.\e[0m"
        echo "Usage: run_pretrain [PRESET] [EPOCHS] [MODEL] [DATASET] [NOTE] [OPTIONAL_TIME]"
        return 1
    fi

    if ! type "$target_preset" &> /dev/null; then
        echo -e "\e[1;31mPreset '$target_preset' does not exist.\e[0m"
        return 1
    fi

    local job_name="${epochs}_${model}_${dataset}"
    
    local SLURM_TIME="7-00:00:00"

    if [ -n "$opt_time" ]; then
        SLURM_TIME="$opt_time"
    fi

    local cmd="

ARCH=\$(uname -m)
if [[ \"\$ARCH\" == \"aarch64\" || \"\$ARCH\" == \"arm64\" ]]; then
    ENV_PATH=\"/home/golem/scratch/chans/lincsv3/aarch64\"
else
    ENV_PATH=\"/home/golem/scratch/chans/lincsv3\"
fi

echo \"architecture: \$ARCH\"
echo \"environment: \$ENV_PATH\"

cd /home/golem/scratch/chans/lincsv3
julia --project=\"\$ENV_PATH\" -e 'using Pkg; Pkg.instantiate()'

julia --project=\"\$ENV_PATH\" scripts/pretrain/main.jl \
    --modeltype \"$model\" \
    --dataset \"$dataset\" \
    --n_epochs \"$epochs\" \
    --additional_notes \"$note\"
"

    $target_preset "$job_name" "$cmd"
}

training_presets() {
    printf "\n"
    printf "\e[1;36m========================================================================================\e[0m\n"
    printf "\e[1;32mAvailable training presets\e[0m\n"
    printf "\e[1;36m========================================================================================\e[0m\n"

    printf "\e[1;33m%-15s %-10s %-10s %-10s %-18s %-10s\e[0m\n" "Command" "Mode" "Level" "Epochs" "Model" "Dataset"
    printf "%-15s %-10s %-10s %-10s %-18s %-10s\n" "--------------" "---------" "---------" "---------" "------------------" "---------"

    printf "%-15s %-10s %-10s %-10s %-18s %-10s\n" "run_finetune" "e2e, emb" "lvl1, lvl2" "Int" "mlp, rtf, v1, v2" "trt, untrt"
    printf "%-15s %-10s %-10s %-10s %-18s %-10s\n" "run_pretrain" "N/A" "N/A" "Int" "rtf, v1, v2" "trt, untrt"

    printf "\n"
    printf "\e[1;33mFT Usage:\e[0m run_finetune [PRESET] [MODE] [LEVEL] [EPOCHS] [MODEL] [DATASET] [NOTE] \e[90m[TIME]\e[0m\n"
    printf "\e[1;33mPT Usage:\e[0m run_pretrain [PRESET] [EPOCHS] [MODEL] [DATASET] [NOTE] \e[90m[TIME]\e[0m\n"
    printf "\e[90m(Note: TIME is optional. Pass as D-HH:MM:SS. Defaults: FT emb=1d, FT e2e=4d, PT=7d)\e[0m\n"
    printf "\e[1;36m========================================================================================\e[0m\n\n"
}

alias tp="training_presets"
# <<< training runs <<<