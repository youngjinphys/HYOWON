// Native snapshot/catalog schema vocabulary.
// GADGET-like HDF5 names are structural only; HYOWON units and phase-space
// semantics are bound explicitly, and external formats require adapters.
#pragma once

#include "cosmo_nbody/io/artifact_schema.hpp"
#include "cosmo_nbody/io/native_identity.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace cosmo_nbody {
namespace io {
namespace schema {

// Native GADGET-like HDF5 structural names. Semantic interoperability with an
// external code requires an explicit adapter and partner-build validation.

// Groups
inline const std::string GROUP_HEADER = "/Header";
// GADGET-like storage slot for the one effective collisionless cb population;
// this structural name does not imply a separately normalized CDM-only species.
inline const std::string GROUP_PART_TYPE_1 = "/PartType1";
inline const std::string GROUP_PARAMETERS = "/Parameters";
inline const std::string GROUP_CONFIG = "/Config";

// HYOWON native snapshot identity. These are format identity markers, not
// scientific-accuracy claims. Readers fail closed on unknown or missing values.
inline const std::string ATTR_SOFTWARE_NAME{
    native_identity::SOFTWARE_NAME_ATTRIBUTE};
inline constexpr std::string_view VAL_SOFTWARE_NAME =
    native_identity::SOFTWARE_NAME;
inline const std::string ATTR_SNAPSHOT_SCHEMA{artifact_schema::SNAPSHOT_ATTRIBUTE};
inline constexpr std::string_view VAL_SNAPSHOT_SCHEMA =
    artifact_schema::SNAPSHOT;

// Header Attributes
inline const std::string ATTR_NUM_PART_THIS_FILE = "NumPart_ThisFile";
inline const std::string ATTR_NUM_PART_TOTAL = "NumPart_Total";
inline const std::string ATTR_MASS_TABLE = "MassTable";
inline const std::string ATTR_TIME = "Time"; // Scale factor a
inline const std::string ATTR_REDSHIFT = "Redshift";
inline const std::string ATTR_BOX_SIZE = "BoxSize";
inline const std::string ATTR_OMEGA0 = "Omega0";
inline const std::string ATTR_OMEGA_LAMBDA = "OmegaLambda";
inline const std::string ATTR_HUBBLE_PARAM = "HubbleParam";
inline const std::string ATTR_NUM_FILES = "NumFilesPerSnapshot";
inline const std::string ATTR_FLAG_DOUBLE_PRECISION = "Flag_DoublePrecision";
inline const std::string ATTR_UNIT_LENGTH_IN_CM = "UnitLength_in_cm";
inline const std::string ATTR_UNIT_MASS_IN_G = "UnitMass_in_g";
inline const std::string ATTR_UNIT_VELOCITY_IN_CM_PER_S =
    "UnitVelocity_in_cm_per_s";
inline const std::string ATTR_PHYSICS_FINGERPRINT = "PhysicsFingerprint";
inline const std::string ATTR_SEED = "Seed";
inline const std::string ATTR_PARTICLES_PER_DIM = "ParticlesPerDim";
inline const std::string ATTR_PM_MESH_PER_DIM = "PMMeshPerDim";
inline const std::string ATTR_IC_MESH_PER_DIM = "ICMeshPerDim";
inline const std::string ATTR_IC_AMPLITUDE_MODE = "ICAmplitudeMode";
inline const std::string ATTR_IC_PHASE_PAIRING = "ICPhasePairing";
inline const std::string ATTR_SOFTENING_COMOVING = "SofteningComoving";
inline const std::string ATTR_SIGMA8 = "Sigma8";
inline const std::string ATTR_N_S = "n_s";
inline const std::string ATTR_OMEGA_B = "OmegaB";
inline const std::string ATTR_LPT_ORDER = "LptOrder";
inline const std::string ATTR_POWER_SPECTRUM_FIDELITY = "PowerSpectrumFidelity";
inline constexpr std::size_t POWER_SPECTRUM_FIDELITY_MAX_BYTES = 24U;

// Exact native phase-space semantics. These are deliberately independent of the
// source linear-density gauge: generated P(k) inputs bind that gauge separately,
// while an evolved particle snapshot is interpreted as Newtonian N-body phase
// space. Exact-IC ingestion requires every binding and rejects unknown values.
struct SnapshotSemanticBinding {
    std::string_view attribute;
    std::string_view value;
};

// Shared native semantic vocabulary. Each literal has one authority here and is
// reused by snapshot bindings, per-dataset unit labels, and derived catalogs.
// This prevents a writer/reader pair from silently diverging when a semantic
// spelling changes while preserving the existing on-disk bytes.
inline constexpr std::string_view NATIVE_ATTR_PARTICLE_SPECIES =
    "ParticleSpecies";
inline constexpr std::string_view NATIVE_ATTR_PHASE_SPACE_GAUGE =
    "PhaseSpaceGauge";
inline constexpr std::string_view NATIVE_ATTR_MASS_NORMALIZATION =
    "MassNormalization";
inline constexpr std::string_view NATIVE_ATTR_COORDINATE_UNIT =
    "CoordinateUnit";
inline constexpr std::string_view NATIVE_ATTR_VELOCITY_UNIT =
    "VelocityUnit";
inline constexpr std::string_view NATIVE_ATTR_MASS_UNIT = "MassUnit";
inline constexpr std::string_view NATIVE_ATTR_EPOCH_CONVENTION =
    "EpochConvention";
inline constexpr std::string_view NATIVE_ATTR_PERIODIC_BOUNDARY_CONVENTION =
    "PeriodicBoundaryConvention";

// One collisionless fluid represents the mass-weighted cold+baryon density.
// omega_b is not an independently evolved particle or hydrodynamic component.
inline constexpr std::string_view VAL_PARTICLE_SPECIES =
    "cold_plus_baryon_collisionless_mnu0";
inline constexpr std::string_view VAL_PARTICLE_INTERPRETATION =
    "effective_one_fluid_total_matter_proxy";
inline constexpr std::string_view VAL_PARTICLE_MASS_NORMALIZATION =
    "omega_m_rho_crit0_box_volume";
inline constexpr std::string_view VAL_BARYONS_SEPARATELY_EVOLVED = "false";
inline constexpr std::string_view VAL_PHASE_SPACE_GAUGE = "newtonian_nbody";
inline constexpr std::string_view VAL_COORDINATE_UNIT = "Mpc_per_h_comoving";
inline constexpr std::string_view VAL_VELOCITY_UNIT = "km_per_s_peculiar";
inline constexpr std::string_view VAL_MASS_UNIT = "1e10_Msun_per_h";
inline constexpr std::string_view VAL_NATIVE_EPOCH_CONVENTION =
    "scale_factor_a_and_redshift_1_over_a_minus_1";
inline constexpr std::string_view VAL_PERIODIC_BOUNDARY_CONVENTION =
    "wrapped_half_open_0_L";

inline constexpr std::array<SnapshotSemanticBinding, 8>
    NATIVE_SNAPSHOT_SEMANTIC_BINDINGS{{
        {NATIVE_ATTR_PARTICLE_SPECIES, VAL_PARTICLE_SPECIES},
        {NATIVE_ATTR_PHASE_SPACE_GAUGE, VAL_PHASE_SPACE_GAUGE},
        {NATIVE_ATTR_MASS_NORMALIZATION, VAL_PARTICLE_MASS_NORMALIZATION},
        {NATIVE_ATTR_COORDINATE_UNIT, VAL_COORDINATE_UNIT},
        {NATIVE_ATTR_VELOCITY_UNIT, VAL_VELOCITY_UNIT},
        {NATIVE_ATTR_MASS_UNIT, VAL_MASS_UNIT},
        {NATIVE_ATTR_EPOCH_CONVENTION, VAL_NATIVE_EPOCH_CONVENTION},
        {NATIVE_ATTR_PERIODIC_BOUNDARY_CONVENTION,
         VAL_PERIODIC_BOUNDARY_CONVENTION},
    }};
inline constexpr std::size_t NATIVE_SNAPSHOT_SEMANTIC_BINDING_COUNT =
    NATIVE_SNAPSHOT_SEMANTIC_BINDINGS.size();

// PartType1 Datasets
inline const std::string DATASET_COORDINATES = "Coordinates";
inline const std::string DATASET_VELOCITIES = "Velocities";
inline const std::string DATASET_PARTICLE_IDS = "ParticleIDs";
inline const std::string DATASET_MASSES = "Masses"; // Only if mass is non-uniform

// Custom Metadata Attributes
inline const std::string ATTR_VELOCITY_CONVENTION = "VelocityConvention";
inline const std::string VAL_VELOCITY_CONVENTION = "v_pec_kms";
inline const std::string VAL_COORDINATE_DATASET_UNIT{VAL_COORDINATE_UNIT};
inline const std::string VAL_VELOCITY_DATASET_UNIT{VAL_VELOCITY_UNIT};
inline const std::string VAL_PARTICLE_ID_DATASET_UNIT =
    "dimensionless_stable_particle_id";
inline const std::string VAL_MASS_DATASET_UNIT{VAL_MASS_UNIT};

inline const std::string ATTR_PRODUCT_LINEAGE = "ProductLineage";
inline const std::string ATTR_MASS_DEFINITION = "MassDefinition";
inline const std::string ATTR_LINKING_LENGTH_B = "LinkingLengthB";
inline const std::string ATTR_MIN_PARTICLES = "MinParticles";
inline const std::string ATTR_SCALE_FACTOR = "ScaleFactor";
inline const std::string ATTR_RUN_METADATA_JSON = "RunMetadataJson";
inline const std::string ATTR_UNIT = "Unit";
inline const std::string ATTR_MOMENTUM_CONVENTION = "MomentumConvention";

inline const std::string VAL_FOF_MASS_DEFINITION =
    "sum_of_member_particle_masses;friends_of_friends_not_spherical_overdensity";

// Split FoF analysis container.
// /FoFMembership owns connectivity only; /HaloDerivedProperties owns every
// quantity computed from member particle phase space.
inline const std::string GROUP_FOF_MEMBERSHIP = "/FoFMembership";
inline const std::string GROUP_HALO_DERIVED_PROPERTIES =
    "/HaloDerivedProperties";
inline const std::string ATTR_MEMBERSHIP_GROUP_PATH = "MembershipGroupPath";
inline const std::string VAL_FOF_MEMBERSHIP_GROUP_PATH = "/FoFMembership";
inline const std::string VAL_FOF_MEMBERSHIP_TRACER_SPECIES =
    "fof_particle_memberships_from_cold_plus_baryon_collisionless_mnu0";
inline const std::string VAL_FOF_MEMBERSHIP_SELECTION_FUNCTION =
    "all_connected_components_above_fof_min_particles";
inline const std::string VAL_FOF_MEMBERSHIP_NORMALIZATION_CONVENTION =
    "stable_particle_id_connected_components";
inline const std::string VAL_HALO_DERIVED_TRACER_SPECIES =
    "fof_halo_properties_from_cold_plus_baryon_collisionless_mnu0";
inline const std::string VAL_HALO_DERIVED_SELECTION_FUNCTION =
    "one_property_row_per_fof_membership";
inline const std::string VAL_HALO_DERIVED_NORMALIZATION_CONVENTION =
    "member_mass_weighted_snapshot_phase_space";

inline const std::string DATASET_MEMBERSHIP_GROUP_IDS = "GroupIDs";
inline const std::string DATASET_MEMBERSHIP_GROUP_SIZES = "GroupSizes";
inline const std::string DATASET_MEMBERSHIP_GROUP_OFFSETS = "GroupOffsets";
inline const std::string DATASET_MEMBERSHIP_PARTICLE_IDS = "MemberParticleIDs";

inline const std::string DATASET_DERIVED_HALO_IDS = "HaloIDs";
inline const std::string DATASET_DERIVED_PARTICLE_COUNTS = "ParticleCounts";
inline const std::string DATASET_DERIVED_MASSES = "Masses";
inline const std::string DATASET_DERIVED_CENTERS = "Centers";
inline const std::string DATASET_DERIVED_MEAN_MOMENTA = "MeanMomenta";
inline const std::string DATASET_DERIVED_MEAN_VELOCITIES = "MeanVelocities";
inline const std::string DATASET_DERIVED_VELOCITY_DISPERSIONS =
    "VelocityDispersions";
inline const std::string DATASET_DERIVED_VELOCITY_DISPERSION_3D =
    "VelocityDispersion3D";
inline const std::string DATASET_DERIVED_ANGULAR_MOMENTA_COMOVING =
    "AngularMomentaComoving";
inline const std::string DATASET_DERIVED_SPECIFIC_ANGULAR_MOMENTA_COMOVING =
    "SpecificAngularMomentaComoving";
inline const std::string DATASET_DERIVED_RMS_RADII_COMOVING =
    "RmsRadiiComoving";
inline const std::string DATASET_DERIVED_MAX_RADII_COMOVING =
    "MaxRadiiComoving";

inline const std::string ATTR_CATALOG_BOX_SIZE = "BoxSizeMpcPerH";
inline const std::string ATTR_SELECTION_FUNCTION = "SelectionFunction";
inline const std::string ATTR_COORDINATE_FRAME = "CoordinateFrame";
inline const std::string ATTR_PERIODIC_BOUNDARY_CONVENTION{
    NATIVE_ATTR_PERIODIC_BOUNDARY_CONVENTION};
inline const std::string ATTR_EPOCH_CONVENTION{NATIVE_ATTR_EPOCH_CONVENTION};
inline const std::string ATTR_WEIGHTING_CONVENTION = "WeightingConvention";
inline const std::string ATTR_TRACER_SPECIES = "TracerSpecies";
inline const std::string ATTR_PHASE_SPACE_GAUGE{NATIVE_ATTR_PHASE_SPACE_GAUGE};
inline const std::string ATTR_NORMALIZATION_CONVENTION =
    "NormalizationConvention";
inline const std::string VAL_CATALOG_COORDINATE_FRAME =
    "comoving_cartesian";
inline const std::string VAL_CATALOG_PERIODIC_BOUNDARY_CONVENTION{
    VAL_PERIODIC_BOUNDARY_CONVENTION};
inline const std::string VAL_CATALOG_EPOCH_CONVENTION =
    "catalog_scale_factor_a";
inline const std::string VAL_CATALOG_PHASE_SPACE_GAUGE{VAL_PHASE_SPACE_GAUGE};
inline const std::string VAL_CATALOG_WEIGHTING_CONVENTION =
    "unit_weight_per_catalog_row";

inline const std::string UNIT_DIMENSIONLESS = "dimensionless";
inline const std::string UNIT_PARTICLES = "particles";
inline const std::string UNIT_CODE_MASS = "code_mass";
inline const std::string UNIT_COMOVING_MPC_PER_H{VAL_COORDINATE_UNIT};
inline const std::string UNIT_CANONICAL_MOMENTUM = "a_times_v_pec";
inline const std::string UNIT_PECULIAR_VELOCITY = "v_pec";
inline const std::string UNIT_COMOVING_ANGULAR_MOMENTUM =
    "code_mass_times_Mpc_per_h_comoving_times_v_pec";
inline const std::string UNIT_SPECIFIC_COMOVING_ANGULAR_MOMENTUM =
    "Mpc_per_h_comoving_times_v_pec";
inline const std::string VAL_MOMENTUM_CONVENTION = "p_equals_a_times_v_pec";
inline const std::string VAL_FOF_VELOCITY_CONVENTION = "v_pec_equals_p_over_a";
inline const std::string VAL_ANGULAR_MOMENTUM_CONVENTION =
    "sum_m_dx_comoving_cross_vpec_minus_mean_vpec";

} // namespace schema
} // namespace io
} // namespace cosmo_nbody
