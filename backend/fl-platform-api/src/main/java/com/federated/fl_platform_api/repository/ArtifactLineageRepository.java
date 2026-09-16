package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.ArtifactLineage;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.UUID;

/** Provenance DAG edges. findByChildId walks toward parents (base/predecessor). */
public interface ArtifactLineageRepository extends JpaRepository<ArtifactLineage, UUID> {
    List<ArtifactLineage> findByChildId(UUID childId);
    List<ArtifactLineage> findByParentId(UUID parentId);
}
