#!/usr/bin/env bash
#
# run julia script
# options for julia can be provided by ,e.g., ElemCo.sh @j "-p 10" [other options for the script]
#

MAIN="src/ElemCo.jl"
JULIA="julia"
JULIA_OPTIONS=

if [ $# -gt 1 ]; then
  if [[ $1 == "@j" ]]; then
    shift
    JULIA_OPTIONS=$1
    shift
  fi
fi

UNAME=$(uname -s)
BASE=$PWD
if [[ "$UNAME" -eq "Linux" ]]; then
   BASE=$(dirname $(readlink -f $0))
elif [[ "$UNAME" -eq "Darwin" ]]; then
  # uses greadlink from coreutils
  BASE=$(dirname $(greadlink -f $0))
fi
#echo $BASE

$JULIA $JULIA_OPTIONS $BASE/$MAIN "$@"

