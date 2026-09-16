package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.ArtifactBlob;
import org.springframework.data.jpa.repository.JpaRepository;

/** Content-addressed blob metadata, keyed by sha256 (the dedup layer). */
public interface ArtifactBlobRepository extends JpaRepository<ArtifactBlob, String> {
}
