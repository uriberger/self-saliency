#!/bin/bash
# Cluster-portable resolution of the two settings that differ between the machines this
# repo runs on. SOURCE this (do not execute it) from a launcher, after REPO is set:
#
#     source "$REPO/cluster_env.sh"
#     PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
#     sr1_find_submit_job || { echo "ERROR: submit_job not found" >&2; exit 1; }
#
# Resolve PARTITION *after* DURATION is final (i.e. after argument parsing, not next to
# the defaults): which partitions can hold the job depends on how long it asks for, and a
# list picked against the wrong length silently drops the best pool -- see SR1_JOB_HOURS.
#
# Why this exists: the origin cluster and oci-hsg-cs-001 (GB200/aarch64) disagree on
#
#   * SLURM partition names -- origin has batch_singlenode / batch_block1; oci-hsg-cs-001
#     has batch (4h) / batch_long (7d) / cpu / cpu_datamover. Submitting to a partition that
#     does not exist is not a clean failure: the job is simply never schedulable.
#   * ADLR cluster-interface release -- the launchers pinned
#     21.1_2026-04-15_21-25-57, which exists on the origin but not on oci-hsg-cs-001
#     (which has 20.0, 21.0, 21.2, 21.3 and a 'latest' symlink).
#
# Both are resolved by asking the machine rather than branching on a cluster name, so a
# third cluster works with no edit. An explicit PARTITION=... in the environment always
# wins -- these only supply the default.

# The preference list used when sr1_pick_partition is called with no arguments. Every
# launcher in this repo submits ONE node with 8 GPUs, so every name here must be able to
# hold that shape; the LENGTH they ask for is no longer fixed, which is what SR1_JOB_HOURS
# and sr1__drop_too_short below are for. Ordered least-contended-first, but the order is
# only a tie-break: SLURM considers all of them and takes whichever can start the job
# soonest.
#
# Why this is a list and not one name: pinning a single name is what left 8-GPU jobs queued
# for a day behind a pool the scheduler had no reason to prefer. The clusters get there by
# different routes, and the list has to cover both:
#
#   * polar -- the pools differ by orders of magnitude in SIZE, and the small ones are
#     shared with the interactive pool.
#   * oci-nrt-cs-001 -- size says nothing, because every GPU partition is backed by the SAME
#     554 nodes. What differs is QUEUE DEPTH. batch_block1 is Default=YES and so collects
#     nearly every job on the cluster (~880 pending); batch_singlenode is the same hardware
#     with ~3. Measured with `sbatch --test-only`, that was a 13.5-hour difference in
#     predicted start for one identical 8-GPU job. Both are listed and both earn their
#     place: batch_singlenode is usually empty, and batch_block1 is PriorityTier=20 against
#     batch_singlenode's 10, so it wins outright whenever it does have room.
#
# batch_short is listed FIRST among the oci names, and it is the reason SR1_JOB_HOURS
# exists. It caps at MaxTime=2h, so it is invisible to a 4h job -- which is what these
# launchers used to ask for, and why it was excluded here. Once the GRPO run dropped to 1h
# chunks it became reachable, and it earns its place on two things (measured 2026-08-24
# with cluster_capacity_report.sh, 1 node x 8 GPUs):
#
#   * PriorityTier=40, against batch_block1's 20 and batch_singlenode's 10 -- the highest
#     of any GPU partition open to this account. With PreemptType=preempt/partition_prio
#     that also means a batch_short job PREEMPTS the ~240 running backfill jobs (tier 1,
#     PreemptMode=CANCEL) rather than waiting behind them, while being unpreemptable
#     itself (PreemptMode=OFF).
#   * `sbatch --test-only` at 1h: batch_short and batch_singlenode both start within a
#     minute, batch_block1 in ~6.7 h. The same probe at 4h is a hard rejection from
#     batch_short ("Requested time limit is invalid"), which is exactly the failure
#     sr1__drop_too_short exists to keep out of the list.
#
# What it is NOT is uniformly faster, and the honest numbers should stay here so nobody
# reorders the list on a hunch. Over 24h of 8-GPU jobs that actually started, batch_short's
# median wait was 1.1 h (138 jobs, 32% under 5 min) against batch_block1's ~0 (190 jobs,
# 75% under 5 min). batch_block1's real hazard is not the wait of the jobs that start, it
# is the 860-deep 8-GPU backlog with a mean age of ~16 DAYS that you might join instead;
# batch_short's equivalent backlog is 23 deep at ~18 h.
#
# That asymmetry is fine because this is a UNION, not a choice: SLURM considers every name
# in the list and starts the job wherever it can start soonest, so an extra eligible pool
# is upside with no downside -- provided it cannot PARK the job.
#
# On the parking risk, and this is the one thing to check before trusting the above: this
# cluster caps GPUS-per-user, not JOBS-per-user. MaxJobsPU is empty for every GPU QoS
# here, so sr1__drop_user_capped -- which only reads MaxJobsPU -- has nothing to say about
# batch_short either way. The live limit is its QoS 4_nodes_per_user_20_nodes_max:
# MaxTRESPU=gres/gpu=32 (4 nodes) for this user, GrpTRES=gres/gpu=160 (20 nodes) for the
# whole partition, DenyOnLimit. Sitting at a MaxTRESPU parks a multi-partition job exactly
# the way MaxJobsPU does, just with Reason=QOSMaxGRESPerUser -- which is how `interactive`
# (cap 16, and two 8-GPU shells already spend it) parks jobs today.
#
# 32 GPUs is far from what we spend there: one 8-GPU trainer plus 1-GPU bench evals. But
# if the whole list ever parks with StartTime=Unknown and Reason=QOSMax*, this is the
# limit to check first, and dropping batch_short is the fix.
#
# Names still NOT added for oci-nrt-cs-001, so nobody re-derives this: interactive (its QoS
# caps a user at 16 GPUs, and two interactive shells already spend exactly that -- see
# sr1__drop_user_capped for why a per-user cap parks the whole job), backfill
# (PreemptMode=CANCEL at PriorityTier=1, so a training job is killed rather than requeued,
# and sr1__drop_backfill_mix would strip it from a mixed list anyway), and admin /
# batch_large / sniff-gpu-nodes (AllowAccounts excludes us). batch_long is likewise closed
# to this account here, but it stays for oci-hsg-cs-001; alongside a partition that does
# work it is simply ignored, because this cluster sets EnforcePartLimits=ANY.
SR1_DEFAULT_PARTITIONS=${SR1_DEFAULT_PARTITIONS:-"polar4 polar3 polar batch_short batch_singlenode batch_block1 batch_long batch"}

# How many hours the job about to be submitted asks for. sr1_pick_partition drops any
# partition whose MaxTime is shorter than this, so a list can name pools that only suit
# short jobs (batch_short) without a long job ever trying to use them.
#
# The default is 4 because that is what every launcher here asked for before the GRPO run
# moved to 1h chunks: unset, the resolved list is exactly what it was. Launchers set it
# from their own DURATION, so `DURATION=4 bash launch_...` correctly gives up batch_short
# and `DURATION=1` correctly gains it.
SR1_JOB_HOURS=${SR1_JOB_HOURS:-4}

# sr1_pick_partition [<preferred> ...]
# Echo a comma-separated list of the named partitions that actually exist here, in the
# order given -- the form `sbatch --partition` and `submit_job --partition` both accept.
# With no arguments, use $SR1_DEFAULT_PARTITIONS. If none of the names match, echo the
# cluster's default partition (the one sinfo marks with '*'), because any real partition
# beats a name that cannot be scheduled.
#
# Three filters are then applied; see the helpers below for why. All of them fail open,
# and the last two are additionally skipped when only one partition survives.
sr1_pick_partition() {
    local avail want default list=
    [ $# -eq 0 ] && set -- $SR1_DEFAULT_PARTITIONS
    avail=$(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | sort -u)
    if [ -z "$avail" ]; then echo "${1:-batch}"; return 0; fi   # no SLURM here (e.g. a login shell without it)
    for want in "$@"; do
        printf '%s\n' "$avail" | grep -qx -- "$want" && list="${list:+$list }$want"
    done
    if [ -z "$list" ]; then
        default=$(sinfo -h -o '%P' 2>/dev/null | grep -m1 '\*' | tr -d '*')
        echo "${default:-$(printf '%s\n' "$avail" | head -1)}"
        return 0
    fi
    list=$(sr1__drop_too_short "$list")
    list=$(sr1__drop_backfill_mix "$list")
    list=$(sr1__drop_user_capped "$list")
    printf '%s\n' "$list" | tr ' ' ','
}

# sr1__slurm_minutes <slurm time string>
# SLURM prints times as [days-]HH:MM:SS with the leading fields omitted, so "30:00" is
# thirty MINUTES and "04:00:00" is four hours. Echo the value in minutes; UNLIMITED,
# INFINITE, NONE and anything unparseable come back as a number large enough that no
# caller treats the partition as too small.
sr1__slurm_minutes() {
    local t="$1" d=0 a b c
    case "$t" in
        *-*) d=${t%%-*}; t=${t#*-} ;;
    esac
    IFS=: read -r a b c <<< "$t"
    # Anything that is not a plain integer here (UNLIMITED, INFINITE, NONE, an empty
    # field, a format we have not seen) is "big enough" -- never a reason to drop a pool.
    case "$d" in ''|*[!0-9]*) echo 999999; return 0 ;; esac
    case "$a" in ''|*[!0-9]*) echo 999999; return 0 ;; esac
    case "$t" in
        *:*:*) case "$b" in ''|*[!0-9]*) echo 999999; return 0 ;; esac
               echo $(( 10#$d * 1440 + 10#$a * 60 + 10#$b )) ;;   # HH:MM:SS
        *)     echo $(( 10#$d * 1440 + 10#$a )) ;;                # MM:SS, or bare minutes
    esac
}

# sr1__drop_too_short <space-separated partitions>
# Drop partitions whose MaxTime is shorter than the $SR1_JOB_HOURS the job asks for.
#
# This is a hard eligibility test, not a preference: `sbatch -p batch_short -t 04:00:00`
# is refused outright with "Requested time limit is invalid (missing or exceeds some
# limit)". This cluster sets EnforcePartLimits=ANY, so naming such a partition alongside
# one that does fit is merely ignored rather than fatal -- but on a cluster set to ALL the
# whole submission is rejected, and either way a name that cannot run the job is noise.
#
# Filtering here rather than hardcoding the list is what lets batch_short be listed at all:
# the same list then serves the 1h GRPO chunks (which gain it) and the 4h launchers (which
# silently give it up). Fails open on a partition whose MaxTime cannot be read, and leaves
# the list alone if the filter would empty it -- an unschedulable list with a clear SLURM
# error beats a silently different one.
sr1__drop_too_short() {
    local list="$1" kept= dropped= p want maxtimes limit hours="${SR1_JOB_HOURS:-4}"
    # A non-integer (or zero) length is not something to guess at: leave every name in.
    case "$hours" in ''|*[!0-9]*|0) printf '%s\n' "$list"; return 0 ;; esac
    want=$(( 10#$hours * 60 ))
    maxtimes=$(scontrol show partition -o 2>/dev/null |
        sed -n 's/^PartitionName=\([^ ]*\).*[^A-Za-z]MaxTime=\([^ ]*\).*/\1 \2/p')
    [ -n "$maxtimes" ] && for p in $list; do
        limit=$(printf '%s\n' "$maxtimes" | awk -v p="$p" '$1 == p {print $2; exit}')
        if [ -n "$limit" ] && [ "$(sr1__slurm_minutes "$limit")" -lt "$want" ]
        then dropped="${dropped:+$dropped }$p($limit)"
        else kept="${kept:+$kept }$p"; fi
    done
    [ -z "$kept" ] && { printf '%s\n' "$list"; return 0; }
    [ -n "$dropped" ] && echo "sr1_pick_partition: dropped $dropped (MaxTime below the ${SR1_JOB_HOURS}h this job asks for)" >&2
    printf '%s\n' "$kept"
}

# sr1__drop_backfill_mix <space-separated partitions>
# SLURM here rejects a submission that names both backfill and non-backfill partitions
# ("Submitting to a mix of backfill and non-backfill partitions is not allowed"), drops the
# backfill ones and warns. Do it ourselves so the warning does not scroll past on every
# launch. A backfill-only list is a legal submission, so that is left alone.
sr1__drop_backfill_mix() {
    local list="$1" kept= dropped= p
    case "$list" in *backfill*) ;; *) printf '%s\n' "$list"; return 0 ;; esac
    for p in $list; do
        case "$p" in backfill*) dropped="${dropped:+$dropped }$p" ;;
                     *)         kept="${kept:+$kept }$p" ;; esac
    done
    [ -z "$kept" ] && { printf '%s\n' "$list"; return 0; }
    echo "sr1_pick_partition: dropped $dropped (SLURM refuses to mix backfill and non-backfill)" >&2
    printf '%s\n' "$kept"
}

# sr1__drop_user_capped <space-separated partitions>
# Drop partitions whose QoS sets a per-user running-job cap (MaxJobsPerUser).
#
# This one is not obvious and it bites hard. QOSMaxJobsPerUserLimit is a JOB-level hold, not
# a per-partition one: if you are at the cap for ONE partition in the list, the job parks
# across ALL of them and stops being considered anywhere -- Reason flips from Priority to
# QOSMaxJobsPerUserLimit and StartTime goes Unknown. A cap of one running job is the
# pathological case, since a single shell you already hold is enough to trigger it.
#
# Such a partition is still perfectly good on its own; reach it with an explicit
# PARTITION=<name>, where there is no other partition to park.
sr1__drop_user_capped() {
    local list="$1" kept= dropped= p capped
    case "$list" in *" "*) ;; *) printf '%s\n' "$list"; return 0 ;; esac   # single partition: no parking risk
    capped=$(
        { scontrol show partition -o 2>/dev/null |
              sed -n 's/^PartitionName=\([^ ]*\).*[^A-Za-z]QoS=\([^ ]*\).*/\1 \2/p'
          echo '--'
          sacctmgr -n -P show qos format=Name,MaxJobsPU 2>/dev/null | awk -F'|' '$2 != "" {print $1}'
        } | awk '/^--$/{s=1;next} s{cap[$1]=1;next} {pq[$1]=$2}
                 END{for (p in pq) if (cap[pq[p]]) print p}'
    )
    [ -z "$capped" ] && { printf '%s\n' "$list"; return 0; }   # no sacctmgr, or no caps anywhere: fail open
    for p in $list; do
        if printf '%s\n' "$capped" | grep -qx -- "$p"
        then dropped="${dropped:+$dropped }$p"
        else kept="${kept:+$kept }$p"; fi
    done
    [ -z "$kept" ] && { printf '%s\n' "$list"; return 0; }     # would empty the list: leave it alone
    [ -n "$dropped" ] && echo "sr1_pick_partition: dropped $dropped (per-user QoS job cap parks a multi-partition job)" >&2
    printf '%s\n' "$kept"
}

# sr1_find_submit_job
# Put the ADLR submit_job wrapper on PATH. Prefers an already-visible submit_job, then the
# 'latest' symlink, then the newest versioned release; the tree is mounted under /lustre/fs1
# on some clusters and /lustre/fsw on others. `[ -x ]` follows symlinks, so a broken
# cross-filesystem symlink is skipped rather than selected. Returns non-zero if not found.
sr1_find_submit_job() {
    command -v submit_job >/dev/null 2>&1 && return 0
    local root cand
    for root in \
        /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface \
        /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface; do
        for cand in "$root/latest" $(ls -1dt "$root"/*/ 2>/dev/null); do
            if [ -x "${cand%/}/submit_job" ]; then
                export PATH="${cand%/}:$PATH"
                return 0
            fi
        done
    done
    return 1
}
