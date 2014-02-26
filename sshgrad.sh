#!/usr/bin/expect

set gradlabnum [lrange $argv 0 0]

spawn /usr/bin/ssh -g cuijean@gradlab$gradlabnum.cs.utah.edu
expect "*password:*"
send -- "432@cuijean\r"

interact
