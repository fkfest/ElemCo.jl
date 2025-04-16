"""
    VersionInfo

This module provides information about the version of the package.
"""
module VersionInfo
export devel, version, git_hash

"""
    devel()

Return true if the version is a development version.
"""
devel() = false
const __VERSION__ = "0.14.0" * (devel() ? "+" : "")


"""
    version()

Return the version of the package.
"""
version() = __VERSION__

"""
    git_hash()

Return the git hash of the current commit, if available.
"""
function git_hash()
  srcpath = @__DIR__
  if isdir(joinpath(srcpath,"..",".git"))
    # get hash from git
    try
      hash = read(`git -C $srcpath rev-parse HEAD`, String)
      return hash[1:end-1]
    catch
      # get hash from .git/HEAD
      try
        head = read(joinpath(srcpath,"..",".git","HEAD"), String)
        head = split(head)[2]
        hash = read(joinpath(srcpath,"..",".git",head), String)
        return hash[1:end-1]
      catch
        return "unknown"
      end
    end
  end
end

end # module
