"""
    Abstract types to resolve circular dependencies
"""
module AbstractEC
export AbstractECInfo
export AbstractDeterminant

abstract type AbstractECInfo end

"""
    AbstractDeterminant

Abstract type for determinants with alpha and beta occupation patterns.
Concrete implementations should have `alpha` and `beta` fields.
"""
abstract type AbstractDeterminant end

end #module