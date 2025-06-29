# Niri

My private fork of `niri` to patch in features that I wouldn't be able to live
without.

Note: Since this is a soft-fork on `niri`, I might intermitendly rebase on
upstream and therefore force-push (-with-lease of cause) to this branch.
Make sure to consider this when `git pull`ing this repo.

## Features

Primary features are
- Support for submaps using the `submap-set` action
- Support for specifying multiple action inside of one binding
- Some helper functions/cli options to aid me in the development 
For a full list I recommend looking at the commits made on top of upstream: [`git log main..fork/main`](https://github.com/YaLTeR/niri/compare/main...titaniumtraveler:niri:fork/main)

## Why are these features no upstreamed yet?

None of the features are in a state to be upstreamed:

- The submaps are missing support for being set *and* queried via `niri-ipc`.
  A complete implementation would also need to support communicating submap
  updates via event stream.

- Multiactions are not supported in the IPC and the project might not want to go
  that step for performance reasons.

- The hotkey-ui is only marginaly supported and only so far as it is necessary for
  the code to compile.
  I don't use it basically at all, so I didn't invest the time yet to adapt the
  code to make it support both having multiple actions associated with a bind
  *and* being able to present any submap other than the default one.

- Documentation of these features.

## Can I run this on my machine?

I wouldn't necessarily recommend it. *I* do and it *seems* to pass all tests,
but I didn't write any tests specifically for these features, so there might be
hidden bugs somewhere in there.
In the end I made this fork to fulfill my needs and it does that.

For install instructions see the upstream project for how to install `niri` from
source.
