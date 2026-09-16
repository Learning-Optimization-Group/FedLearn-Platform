package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.ArtifactDto;
import com.federated.fl_platform_api.service.MarketplaceService;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

/**
 * FE-12: the org-internal adapter marketplace surface.
 *
 * <ul>
 *   <li>{@code GET /api/marketplace/adapters} — the feed: published LORA_ADAPTERs the caller may see,
 *       newest-published first (org-scoped; a platform admin sees all).</li>
 *   <li>{@code POST /api/marketplace/adapters/{id}/publish} — publish (project owner or admin).</li>
 *   <li>{@code DELETE /api/marketplace/adapters/{id}/publish} — withdraw.</li>
 * </ul>
 *
 * <p>All authorization + org-scope + no-existence-leak rules live in {@link MarketplaceService}.</p>
 */
@RestController
@RequestMapping("/api/marketplace")
public class MarketplaceController {

    private final MarketplaceService marketplace;

    public MarketplaceController(MarketplaceService marketplace) {
        this.marketplace = marketplace;
    }

    @GetMapping("/adapters")
    public List<ArtifactDto> browse() {
        return marketplace.listPublishedAdapters();
    }

    @PostMapping("/adapters/{id}/publish")
    public ArtifactDto publish(@PathVariable UUID id) {
        return marketplace.publish(id);
    }

    @DeleteMapping("/adapters/{id}/publish")
    public ArtifactDto unpublish(@PathVariable UUID id) {
        return marketplace.unpublish(id);
    }
}
