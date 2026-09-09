#include "cosmo_nbody/analysis/halo_membership_catalog.hpp"

#include "cosmo_nbody/io/durable_file_publication.hpp"
#include "cosmo_nbody/io/hdf5_handle.hpp"
#include "cosmo_nbody/io/output_schema.hpp"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cosmo_nbody::analysis {
namespace {

// I/O coordinates only. They do not enter any scientific calculation.
constexpr hsize_t metadata_chunk_items = 4096;
constexpr hsize_t member_chunk_items = 131072;
constexpr std::size_t metadata_batch_items = 4096;
constexpr std::size_t member_batch_target_bytes = std::size_t{4} * 1024 * 1024;

constexpr const char* DATASET_CANDIDATE_IDS = "CandidateIDs";
constexpr const char* DATASET_DEBLENDED_SEED_IDS = "DeblendedSeedIDs";
constexpr const char* DATASET_PEAK_PARTICLE_IDS = "PeakParticleIDs";
constexpr const char* DATASET_GROUP_OFFSETS = "GroupOffsets";
constexpr const char* DATASET_GROUP_SIZES = "GroupSizes";
constexpr const char* DATASET_AVAILABLE = "Available";
constexpr const char* DATASET_MEMBER_PARTICLE_IDS = "MemberParticleIDs";

std::uint64_t checked_u64(std::size_t value, const char* label) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(std::string(label) + " exceeds uint64");
        }
    }
    return static_cast<std::uint64_t>(value);
}

hsize_t checked_hsize(std::uint64_t value, const char* label) {
    if constexpr (sizeof(hsize_t) < sizeof(std::uint64_t)) {
        if (value > static_cast<std::uint64_t>(
                std::numeric_limits<hsize_t>::max())) {
            throw std::overflow_error(std::string(label) + " exceeds hsize_t");
        }
    }
    return static_cast<hsize_t>(value);
}

bool canonical_sha256(std::string_view value) {
    return value.size() == 64U
        && std::all_of(value.begin(), value.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

void write_string_attr(hid_t location, const char* name, std::string_view value) {
    if (value.empty()) {
        throw std::invalid_argument(
            std::string("Empty halo-membership HDF5 attribute: ") + name);
    }
    auto space = io::H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR), std::string("create attribute space ") + name);
    auto type = io::H5TypeHandle::checked(
        H5Tcopy(H5T_C_S1), std::string("copy attribute type ") + name);
    io::check_hdf5(
        H5Tset_size(type.get(), value.size() + 1),
        std::string("set attribute size ") + name);
    io::check_hdf5(
        H5Tset_strpad(type.get(), H5T_STR_NULLTERM),
        std::string("set attribute padding ") + name);
    auto attribute = io::H5AttributeHandle::checked(
        H5Acreate2(
            location, name, type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT),
        std::string("create attribute ") + name);
    const std::string stored(value);
    io::check_hdf5(
        H5Awrite(attribute.get(), type.get(), stored.c_str()),
        std::string("write attribute ") + name);
}

void write_u64_attr(hid_t location, const char* name, std::uint64_t value) {
    auto space = io::H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR), std::string("create attribute space ") + name);
    auto attribute = io::H5AttributeHandle::checked(
        H5Acreate2(
            location,
            name,
            H5T_STD_U64LE,
            space.get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        std::string("create attribute ") + name);
    io::check_hdf5(
        H5Awrite(attribute.get(), H5T_NATIVE_UINT64, &value),
        std::string("write attribute ") + name);
}

void write_bool_attr(hid_t location, const char* name, bool value) {
    const std::uint8_t stored = value ? 1U : 0U;
    auto space = io::H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR), std::string("create attribute space ") + name);
    auto attribute = io::H5AttributeHandle::checked(
        H5Acreate2(
            location,
            name,
            H5T_STD_U8LE,
            space.get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        std::string("create attribute ") + name);
    io::check_hdf5(
        H5Awrite(attribute.get(), H5T_NATIVE_UINT8, &stored),
        std::string("write attribute ") + name);
}

void write_double_attr(hid_t location, const char* name, double value) {
    auto space = io::H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR), std::string("create attribute space ") + name);
    auto attribute = io::H5AttributeHandle::checked(
        H5Acreate2(
            location,
            name,
            H5T_IEEE_F64LE,
            space.get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        std::string("create attribute ") + name);
    io::check_hdf5(
        H5Awrite(attribute.get(), H5T_NATIVE_DOUBLE, &value),
        std::string("write attribute ") + name);
}

io::H5DatasetHandle create_extendible_dataset(
    hid_t group,
    const char* name,
    hid_t file_type,
    hsize_t chunk_items) {
    const hsize_t dims[1] = {0};
    const hsize_t max_dims[1] = {H5S_UNLIMITED};
    auto space = io::H5SpaceHandle::checked(
        H5Screate_simple(1, dims, max_dims),
        std::string("create dataset space ") + name);
    auto properties = io::H5PropertyHandle::checked(
        H5Pcreate(H5P_DATASET_CREATE),
        std::string("create dataset properties ") + name);
    const hsize_t chunk[1] = {chunk_items};
    io::check_hdf5(
        H5Pset_chunk(properties.get(), 1, chunk),
        std::string("set dataset chunking ") + name);
    return io::H5DatasetHandle::checked(
        H5Dcreate2(
            group,
            name,
            file_type,
            space.get(),
            H5P_DEFAULT,
            properties.get(),
            H5P_DEFAULT),
        std::string("create dataset ") + name);
}

void append_dataset(
    hid_t dataset,
    const char* name,
    hid_t memory_type,
    const void* data,
    std::uint64_t old_count,
    std::size_t append_count) {
    if (append_count == 0) return;
    const std::uint64_t add = checked_u64(append_count, name);
    if (old_count > std::numeric_limits<std::uint64_t>::max() - add) {
        throw std::overflow_error(std::string(name) + " extent exceeds uint64");
    }
    const hsize_t new_dims[1] = {checked_hsize(old_count + add, name)};
    io::check_hdf5(
        H5Dset_extent(dataset, new_dims), std::string("extend dataset ") + name);
    auto file_space = io::H5SpaceHandle::checked(
        H5Dget_space(dataset), std::string("open dataset space ") + name);
    const hsize_t start[1] = {checked_hsize(old_count, name)};
    const hsize_t count[1] = {checked_hsize(add, name)};
    io::check_hdf5(
        H5Sselect_hyperslab(
            file_space.get(), H5S_SELECT_SET, start, nullptr, count, nullptr),
        std::string("select append slab ") + name);
    auto memory_space = io::H5SpaceHandle::checked(
        H5Screate_simple(1, count, nullptr),
        std::string("create memory space ") + name);
    io::check_hdf5(
        H5Dwrite(
            dataset,
            memory_type,
            memory_space.get(),
            file_space.get(),
            H5P_DEFAULT,
            data),
        std::string("append dataset ") + name);
}

const halo::SOMassRequest& named_so_request(
    halo::StandardSOMassDefinition definition) {
    static const std::vector<halo::SOMassRequest> requests =
        halo::standard_so_mass_requests();
    const auto request = std::find_if(
        requests.begin(), requests.end(), [&](const halo::SOMassRequest& item) {
            return item.definition == definition;
        });
    if (request == requests.end()) {
        throw std::invalid_argument(
            "Geometric SO membership requires a named mass definition");
    }
    return *request;
}

const char* so_group_name(halo::StandardSOMassDefinition definition) {
    switch (definition) {
        case halo::StandardSOMassDefinition::M200m: return "M200m";
        case halo::StandardSOMassDefinition::M200c: return "M200c";
        case halo::StandardSOMassDefinition::MvirBN98: return "Mvir_BN98";
        case halo::StandardSOMassDefinition::Unknown: break;
    }
    throw std::invalid_argument(
        "Geometric SO membership requires a named mass definition");
}

class MembershipTable {
public:
    MembershipTable(
        hid_t parent,
        const char* group_name,
        std::string_view membership_view,
        std::optional<halo::StandardSOMassDefinition> mass_definition)
        : group_(io::H5GroupHandle::checked(
              H5Gcreate2(
                  parent,
                  group_name,
                  H5P_DEFAULT,
                  H5P_DEFAULT,
                  H5P_DEFAULT),
              std::string("create membership group ") + group_name)),
          candidate_ids_(create_extendible_dataset(
              group_.get(), DATASET_CANDIDATE_IDS, H5T_STD_U64LE,
              metadata_chunk_items)),
          seed_ids_(create_extendible_dataset(
              group_.get(), DATASET_DEBLENDED_SEED_IDS, H5T_STD_U64LE,
              metadata_chunk_items)),
          peak_ids_(create_extendible_dataset(
              group_.get(), DATASET_PEAK_PARTICLE_IDS, H5T_STD_U64LE,
              metadata_chunk_items)),
          offsets_(create_extendible_dataset(
              group_.get(), DATASET_GROUP_OFFSETS, H5T_STD_U64LE,
              metadata_chunk_items)),
          sizes_(create_extendible_dataset(
              group_.get(), DATASET_GROUP_SIZES, H5T_STD_U64LE,
              metadata_chunk_items)),
          available_(mass_definition.has_value()
                  ? create_extendible_dataset(
                        group_.get(), DATASET_AVAILABLE, H5T_STD_U8LE,
                        metadata_chunk_items)
                  : io::H5DatasetHandle{}),
          member_ids_(create_extendible_dataset(
              group_.get(), DATASET_MEMBER_PARTICLE_IDS, H5T_STD_U64LE,
              member_chunk_items)),
          stores_availability_(mass_definition.has_value()) {
        write_string_attr(group_.get(), "MembershipView", membership_view);
        write_string_attr(
            group_.get(),
            "OffsetConvention",
            "zero_based_index_into_this_group_MemberParticleIDs");
        write_string_attr(
            group_.get(),
            "MemberOrdering",
            "ascending_stable_particle_id_within_each_membership");
        write_string_attr(
            member_ids_.get(),
            io::schema::ATTR_UNIT.c_str(),
            io::schema::VAL_PARTICLE_ID_DATASET_UNIT);
        write_string_attr(
            peak_ids_.get(),
            io::schema::ATTR_UNIT.c_str(),
            io::schema::VAL_PARTICLE_ID_DATASET_UNIT);
        write_string_attr(
            candidate_ids_.get(), io::schema::ATTR_UNIT.c_str(),
            io::schema::UNIT_DIMENSIONLESS);
        write_string_attr(
            seed_ids_.get(), io::schema::ATTR_UNIT.c_str(),
            io::schema::UNIT_DIMENSIONLESS);
        write_string_attr(
            offsets_.get(), io::schema::ATTR_UNIT.c_str(),
            io::schema::UNIT_DIMENSIONLESS);
        write_string_attr(
            sizes_.get(), io::schema::ATTR_UNIT.c_str(),
            io::schema::UNIT_PARTICLES);

        if (mass_definition.has_value()) {
            const auto definition = *mass_definition;
            const auto& request = named_so_request(definition);
            write_string_attr(
                group_.get(),
                io::schema::ATTR_MASS_DEFINITION.c_str(),
                halo::standard_so_mass_definition_name(definition));
            write_string_attr(
                group_.get(),
                "ReferenceDensityKind",
                halo::so_reference_density_name(
                    request.reference_density_kind));
            if (request.fixed_overdensity_threshold.has_value()) {
                write_string_attr(
                    group_.get(),
                    "OverdensityThresholdConvention",
                    "fixed_multiple_of_reference_density");
                write_double_attr(
                    group_.get(),
                    "FixedOverdensityThreshold",
                    static_cast<double>(*request.fixed_overdensity_threshold));
            } else {
                write_string_attr(
                    group_.get(),
                    "OverdensityThresholdConvention",
                    "bryan_norman_1998_flat_lambda_cdm_relative_to_critical");
            }
            write_string_attr(
                group_.get(),
                "UnavailableConvention",
                "Available=0,GroupSize=0,no_member_particle_ids");
            write_string_attr(
                available_.get(), io::schema::ATTR_UNIT.c_str(),
                io::schema::UNIT_DIMENSIONLESS);
        }

        candidate_buffer_.reserve(metadata_batch_items);
        seed_buffer_.reserve(metadata_batch_items);
        peak_buffer_.reserve(metadata_batch_items);
        offset_buffer_.reserve(metadata_batch_items);
        size_buffer_.reserve(metadata_batch_items);
        if (stores_availability_) {
            available_buffer_.reserve(metadata_batch_items);
        }
        member_buffer_.reserve(member_batch_capacity());
    }

    void append(
        std::size_t candidate_id,
        std::size_t seed_id,
        core::ParticleId peak_id,
        bool available,
        std::span<const core::ParticleId> members) {
        if (finalized_) {
            throw std::logic_error("Membership table is already finalized");
        }
        if (!stores_availability_ && !available) {
            throw std::invalid_argument(
                "Deblended membership cannot be marked unavailable");
        }
        if ((!available && !members.empty()) || (available && members.empty())) {
            throw std::invalid_argument(
                "Halo membership availability disagrees with member payload");
        }
        if (!std::is_sorted(members.begin(), members.end())
            || std::adjacent_find(members.begin(), members.end())
                != members.end()) {
            throw std::invalid_argument(
                "Halo membership ParticleIDs must be strictly increasing");
        }

        const std::uint64_t size = checked_u64(members.size(), "membership size");
        if (record_count_ == std::numeric_limits<std::uint64_t>::max()
            || member_count_ > std::numeric_limits<std::uint64_t>::max() - size) {
            throw std::overflow_error(
                "Halo membership offset/size exceeds uint64 representation");
        }

        const std::uint64_t candidate = checked_u64(candidate_id, "candidate id");
        const std::uint64_t seed = checked_u64(seed_id, "deblended seed id");
        const std::uint64_t offset = member_count_;
        const std::uint8_t available_value = available ? 1U : 0U;

        if (members.size() >= member_batch_capacity() && !members.empty()) {
            flush();
            append_metadata_row(
                candidate, seed, peak_id, offset, size, available_value);
            append_dataset(
                member_ids_.get(),
                DATASET_MEMBER_PARTICLE_IDS,
                H5T_NATIVE_UINT64,
                members.data(),
                written_member_count_,
                members.size());
            ++written_record_count_;
            written_member_count_ += size;
            ++record_count_;
            member_count_ += size;
            return;
        }

        candidate_buffer_.push_back(candidate);
        seed_buffer_.push_back(seed);
        peak_buffer_.push_back(peak_id);
        offset_buffer_.push_back(offset);
        size_buffer_.push_back(size);
        if (stores_availability_) available_buffer_.push_back(available_value);
        member_buffer_.insert(member_buffer_.end(), members.begin(), members.end());
        ++record_count_;
        member_count_ += size;
        if (candidate_buffer_.size() >= metadata_batch_items
            || member_buffer_.size() >= member_batch_capacity()) {
            flush();
        }
    }

    void finalize() {
        if (finalized_) return;
        flush();
        if (written_record_count_ != record_count_
            || written_member_count_ != member_count_) {
            throw std::logic_error(
                "Halo membership buffered counts diverged before publication");
        }
        write_u64_attr(group_.get(), "MembershipCount", record_count_);
        write_u64_attr(group_.get(), "MemberParticleIDCount", member_count_);
        candidate_ids_.reset();
        seed_ids_.reset();
        peak_ids_.reset();
        offsets_.reset();
        sizes_.reset();
        available_.reset();
        member_ids_.reset();
        group_.reset();
        finalized_ = true;
    }

    std::uint64_t record_count() const noexcept { return record_count_; }
    std::uint64_t member_count() const noexcept { return member_count_; }

private:
    static constexpr std::size_t member_batch_capacity() {
        static_assert(sizeof(core::ParticleId) == sizeof(std::uint64_t));
        return member_batch_target_bytes / sizeof(core::ParticleId);
    }

    void append_metadata_row(
        std::uint64_t candidate,
        std::uint64_t seed,
        core::ParticleId peak,
        std::uint64_t offset,
        std::uint64_t size,
        std::uint8_t available_value) {
        append_dataset(
            candidate_ids_.get(), DATASET_CANDIDATE_IDS, H5T_NATIVE_UINT64,
            &candidate, written_record_count_, 1);
        append_dataset(
            seed_ids_.get(), DATASET_DEBLENDED_SEED_IDS, H5T_NATIVE_UINT64,
            &seed, written_record_count_, 1);
        append_dataset(
            peak_ids_.get(), DATASET_PEAK_PARTICLE_IDS, H5T_NATIVE_UINT64,
            &peak, written_record_count_, 1);
        append_dataset(
            offsets_.get(), DATASET_GROUP_OFFSETS, H5T_NATIVE_UINT64,
            &offset, written_record_count_, 1);
        append_dataset(
            sizes_.get(), DATASET_GROUP_SIZES, H5T_NATIVE_UINT64,
            &size, written_record_count_, 1);
        if (stores_availability_) {
            append_dataset(
                available_.get(), DATASET_AVAILABLE, H5T_NATIVE_UINT8,
                &available_value, written_record_count_, 1);
        }
    }

    void flush() {
        const std::size_t rows = candidate_buffer_.size();
        if (seed_buffer_.size() != rows || peak_buffer_.size() != rows
            || offset_buffer_.size() != rows || size_buffer_.size() != rows
            || (stores_availability_ && available_buffer_.size() != rows)) {
            throw std::logic_error(
                "Halo membership metadata buffers lost row alignment");
        }
        append_dataset(
            candidate_ids_.get(), DATASET_CANDIDATE_IDS, H5T_NATIVE_UINT64,
            candidate_buffer_.data(), written_record_count_, rows);
        append_dataset(
            seed_ids_.get(), DATASET_DEBLENDED_SEED_IDS, H5T_NATIVE_UINT64,
            seed_buffer_.data(), written_record_count_, rows);
        append_dataset(
            peak_ids_.get(), DATASET_PEAK_PARTICLE_IDS, H5T_NATIVE_UINT64,
            peak_buffer_.data(), written_record_count_, rows);
        append_dataset(
            offsets_.get(), DATASET_GROUP_OFFSETS, H5T_NATIVE_UINT64,
            offset_buffer_.data(), written_record_count_, rows);
        append_dataset(
            sizes_.get(), DATASET_GROUP_SIZES, H5T_NATIVE_UINT64,
            size_buffer_.data(), written_record_count_, rows);
        if (stores_availability_) {
            append_dataset(
                available_.get(), DATASET_AVAILABLE, H5T_NATIVE_UINT8,
                available_buffer_.data(), written_record_count_, rows);
        }
        append_dataset(
            member_ids_.get(), DATASET_MEMBER_PARTICLE_IDS, H5T_NATIVE_UINT64,
            member_buffer_.data(), written_member_count_, member_buffer_.size());

        written_record_count_ += checked_u64(rows, "written membership count");
        written_member_count_ += checked_u64(
            member_buffer_.size(), "written member ParticleID count");
        candidate_buffer_.clear();
        seed_buffer_.clear();
        peak_buffer_.clear();
        offset_buffer_.clear();
        size_buffer_.clear();
        available_buffer_.clear();
        member_buffer_.clear();
    }

    io::H5GroupHandle group_;
    io::H5DatasetHandle candidate_ids_;
    io::H5DatasetHandle seed_ids_;
    io::H5DatasetHandle peak_ids_;
    io::H5DatasetHandle offsets_;
    io::H5DatasetHandle sizes_;
    io::H5DatasetHandle available_;
    io::H5DatasetHandle member_ids_;
    bool stores_availability_{false};
    std::vector<std::uint64_t> candidate_buffer_;
    std::vector<std::uint64_t> seed_buffer_;
    std::vector<core::ParticleId> peak_buffer_;
    std::vector<std::uint64_t> offset_buffer_;
    std::vector<std::uint64_t> size_buffer_;
    std::vector<std::uint8_t> available_buffer_;
    std::vector<core::ParticleId> member_buffer_;
    std::uint64_t record_count_{0};
    std::uint64_t member_count_{0};
    std::uint64_t written_record_count_{0};
    std::uint64_t written_member_count_{0};
    bool finalized_{false};
};

} // namespace

class HaloMembershipCatalogWriter::Impl {
public:
    Impl(
        const std::filesystem::path& path,
        const std::filesystem::path& source_snapshot,
        std::string_view source_sha256,
        core::Real scale_factor,
        core::Real fof_linking_length_b,
        std::size_t fof_min_particles,
        std::size_t peak_density_k_neighbors,
        std::size_t deblended_min_particles,
        core::Real peak_saddle_merge_ratio,
        std::string_view fof_membership_product)
        : publication_(
              path,
              "halo membership analysis catalog",
              io::DurableFilePublicationPolicy::RequireAbsent) {
        if (path.empty()
            || source_snapshot.empty()
            || !canonical_sha256(source_sha256)
            || !std::isfinite(scale_factor)
            || scale_factor <= 0.0
            || !std::isfinite(fof_linking_length_b)
            || fof_linking_length_b <= 0.0
            || fof_min_particles == 0U
            || peak_density_k_neighbors == 0U
            || deblended_min_particles == 0U
            || !std::isfinite(peak_saddle_merge_ratio)
            || peak_saddle_merge_ratio < 0.0
            || peak_saddle_merge_ratio > 1.0
            || fof_membership_product.empty()) {
            throw std::invalid_argument(
                "Halo membership catalog provenance is invalid");
        }

        file_ = io::H5FileHandle::checked(
            H5Fcreate(
                publication_.staging_path().string().c_str(),
                H5F_ACC_EXCL,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "create halo membership analysis catalog");

        write_string_attr(file_.get(), "ProductKind", "analysis_halo_membership_views");
        write_string_attr(file_.get(), "SourceSnapshot", source_snapshot.string());
        write_string_attr(file_.get(), "SourceNativeSnapshotObjectSHA256", source_sha256);
        write_double_attr(
            file_.get(), io::schema::ATTR_SCALE_FACTOR.c_str(),
            static_cast<double>(scale_factor));
        write_string_attr(
            file_.get(), io::schema::ATTR_PRODUCT_LINEAGE.c_str(), "derived_analysis");
        write_string_attr(
            file_.get(), io::schema::ATTR_TRACER_SPECIES.c_str(),
            io::schema::VAL_PARTICLE_SPECIES);
        write_string_attr(
            file_.get(), "TracerInterpretation", io::schema::VAL_PARTICLE_INTERPRETATION);
        write_string_attr(
            file_.get(), "MassNormalization", io::schema::VAL_PARTICLE_MASS_NORMALIZATION);
        write_string_attr(
            file_.get(), "BaryonsSeparatelyEvolved", io::schema::VAL_BARYONS_SEPARATELY_EVOLVED);
        write_double_attr(
            file_.get(), "FoFLinkingLengthB",
            static_cast<double>(fof_linking_length_b));
        write_u64_attr(
            file_.get(), "FoFMinParticles",
            checked_u64(fof_min_particles, "FoF minimum particle count"));
        write_u64_attr(
            file_.get(), "PeakDensityKNeighbors",
            checked_u64(peak_density_k_neighbors, "peak-density k-neighbors"));
        write_u64_attr(
            file_.get(), "DeblendedMinParticles",
            checked_u64(deblended_min_particles, "deblended minimum particle count"));
        write_double_attr(
            file_.get(), "PeakSaddleMergeRatio",
            static_cast<double>(peak_saddle_merge_ratio));
        write_string_attr(
            file_.get(),
            "DeblendedMembershipSelection",
            "retained_density_peak_basins_before_named_so_minimum_particle_cut");
        write_string_attr(
            file_.get(),
            "NamedSOSeedSelection",
            "deblended_hosts_meeting_DeblendedMinParticles");
        write_bool_attr(
            file_.get(),
            "NamedSOMembershipsExistOnlyForSelectedDeblendedSeeds",
            true);
        write_string_attr(file_.get(), "FoFCandidateMembershipProduct", fof_membership_product);
        write_bool_attr(file_.get(), "FoFCandidateMembershipDuplicated", false);
        write_bool_attr(file_.get(), "BoundHostExcludingSubstructureAvailable", false);
        write_string_attr(
            file_.get(),
            "DensityPeakPropertiesProduct",
            "analysis_density_peaks_all.csv");
        write_string_attr(
            file_.get(),
            "DeblendedHostPropertiesProduct",
            "analysis_deblended_hosts_all.csv");
        write_string_attr(
            file_.get(),
            "StandardSOPropertiesProduct",
            "analysis_standard_so.csv");
        write_string_attr(
            file_.get(),
            "StandardSODerivedPropertiesProduct",
            "analysis_standard_so_derived.csv");

        deblended_ = std::make_unique<MembershipTable>(
            file_.get(),
            "DeblendedDensityBasin",
            halo::halo_membership_view_name(
                halo::HaloMembershipView::DeblendedDensityBasinAll),
            std::nullopt);

        geometric_so_group_ = io::H5GroupHandle::checked(
            H5Gcreate2(
                file_.get(),
                "GeometricSO",
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "create GeometricSO membership group");
        write_string_attr(
            geometric_so_group_.get(),
            "MembershipView",
            halo::halo_membership_view_name(
                halo::HaloMembershipView::GeometricSOAll));

        const std::array definitions{
            halo::StandardSOMassDefinition::M200m,
            halo::StandardSOMassDefinition::M200c,
            halo::StandardSOMassDefinition::MvirBN98,
        };
        for (std::size_t index = 0; index < definitions.size(); ++index) {
            const auto definition = definitions[index];
            so_tables_[index] = std::make_unique<MembershipTable>(
                geometric_so_group_.get(),
                so_group_name(definition),
                halo::halo_membership_view_name(
                    halo::HaloMembershipView::GeometricSOAll),
                definition);
        }
    }

    void append_deblended(
        std::size_t candidate_id,
        std::size_t seed_id,
        core::ParticleId peak_id,
        std::span<const core::ParticleId> members) {
        require_open();
        deblended_->append(candidate_id, seed_id, peak_id, true, members);
    }

    void append_so(
        std::size_t candidate_id,
        std::size_t seed_id,
        core::ParticleId peak_id,
        halo::StandardSOMassDefinition definition,
        bool available,
        std::span<const core::ParticleId> members) {
        require_open();
        so_table(definition).append(
            candidate_id, seed_id, peak_id, available, members);
    }

    void finalize() {
        require_open();
        deblended_->finalize();
        std::uint64_t total_records = deblended_->record_count();
        std::uint64_t total_members = deblended_->member_count();
        deblended_.reset();
        for (auto& table : so_tables_) {
            table->finalize();
            if (total_records
                    > std::numeric_limits<std::uint64_t>::max()
                        - table->record_count()
                || total_members
                    > std::numeric_limits<std::uint64_t>::max()
                        - table->member_count()) {
                throw std::overflow_error(
                    "Halo membership catalog aggregate count exceeds uint64");
            }
            total_records += table->record_count();
            total_members += table->member_count();
            table.reset();
        }
        geometric_so_group_.reset();
        write_u64_attr(file_.get(), "MembershipRecordCount", total_records);
        write_u64_attr(file_.get(), "MemberParticleIDCount", total_members);
        io::check_hdf5(
            H5Fflush(file_.get(), H5F_SCOPE_GLOBAL),
            "flush halo membership analysis catalog");
        file_.reset();
        publication_.publish_nonempty();
        finalized_ = true;
    }

private:
    void require_open() const {
        if (finalized_ || !file_) {
            throw std::logic_error("Halo membership catalog is already finalized");
        }
    }

    MembershipTable& so_table(halo::StandardSOMassDefinition definition) {
        switch (definition) {
            case halo::StandardSOMassDefinition::M200m:
                return *so_tables_[0];
            case halo::StandardSOMassDefinition::M200c:
                return *so_tables_[1];
            case halo::StandardSOMassDefinition::MvirBN98:
                return *so_tables_[2];
            case halo::StandardSOMassDefinition::Unknown:
                break;
        }
        throw std::invalid_argument(
            "Geometric SO membership requires a named mass definition");
    }

    io::DurableFilePublication publication_;
    io::H5FileHandle file_;
    std::unique_ptr<MembershipTable> deblended_;
    io::H5GroupHandle geometric_so_group_;
    std::array<std::unique_ptr<MembershipTable>, 3> so_tables_;
    bool finalized_{false};
};

HaloMembershipCatalogWriter::HaloMembershipCatalogWriter(
    const std::filesystem::path& path,
    const std::filesystem::path& source_snapshot,
    std::string_view source_sha256,
    core::Real scale_factor,
    core::Real fof_linking_length_b,
    std::size_t fof_min_particles,
    std::size_t peak_density_k_neighbors,
    std::size_t deblended_min_particles,
    core::Real peak_saddle_merge_ratio,
    std::string_view fof_membership_product)
    : impl_(std::make_unique<Impl>(
          path,
          source_snapshot,
          source_sha256,
          scale_factor,
          fof_linking_length_b,
          fof_min_particles,
          peak_density_k_neighbors,
          deblended_min_particles,
          peak_saddle_merge_ratio,
          fof_membership_product)) {}

HaloMembershipCatalogWriter::~HaloMembershipCatalogWriter() = default;

void HaloMembershipCatalogWriter::append_deblended_density_basin(
    std::size_t candidate_id,
    std::size_t deblended_seed_id,
    core::ParticleId peak_particle_id,
    std::span<const core::ParticleId> member_ids) {
    impl_->append_deblended(
        candidate_id, deblended_seed_id, peak_particle_id, member_ids);
}

void HaloMembershipCatalogWriter::append_geometric_so(
    std::size_t candidate_id,
    std::size_t deblended_seed_id,
    core::ParticleId peak_particle_id,
    halo::StandardSOMassDefinition definition,
    bool available,
    std::span<const core::ParticleId> member_ids) {
    impl_->append_so(
        candidate_id,
        deblended_seed_id,
        peak_particle_id,
        definition,
        available,
        member_ids);
}

void HaloMembershipCatalogWriter::finalize() {
    impl_->finalize();
}

} // namespace cosmo_nbody::analysis
