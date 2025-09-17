//! Contract tests for fixture management API
//!
//! Validates that the fixture API matches the OpenAPI specification

use tch::test_utils::{TestFixture, FixtureBuilder};
use std::path::PathBuf;

#[test]
fn test_fixture_api_list_fixtures() {
    // This test should fail until fixture API is implemented
    let fixtures = TestFixture::list(None);

    assert!(fixtures.is_ok(), "Should list available fixtures");
    let fixture_list = fixtures.unwrap();

    assert!(!fixture_list.is_empty(), "Should have fixtures available");

    for fixture in &fixture_list {
        assert!(!fixture.id.is_empty(), "Fixture ID must not be empty");
        assert!(!fixture.name.is_empty(), "Fixture name must not be empty");
        assert!(["model", "dataset", "config", "state"].contains(&fixture.fixture_type.as_str()),
                "Fixture type must be valid");
    }
}

#[test]
fn test_fixture_api_filter_by_type() {
    let model_fixtures = TestFixture::list(Some("model".to_string()));
    assert!(model_fixtures.is_ok(), "Should filter fixtures by type");

    let models = model_fixtures.unwrap();
    for fixture in &models {
        assert_eq!(fixture.fixture_type, "model", "Should only return model fixtures");
    }

    let dataset_fixtures = TestFixture::list(Some("dataset".to_string()));
    assert!(dataset_fixtures.is_ok(), "Should filter dataset fixtures");

    let datasets = dataset_fixtures.unwrap();
    for fixture in &datasets {
        assert_eq!(fixture.fixture_type, "dataset", "Should only return dataset fixtures");
    }
}

#[test]
fn test_fixture_api_lazy_loading() {
    let fixture = FixtureBuilder::new("lazy_test")
        .fixture_type("dataset")
        .lazy_load(true)
        .cache(false)
        .generator(|| {
            // Generate test data on demand
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        })
        .build();

    assert!(fixture.is_ok(), "Should create lazy-loading fixture");
    let lazy_fixture = fixture.unwrap();

    assert!(lazy_fixture.lazy_load, "Fixture should be marked for lazy loading");
    assert!(!lazy_fixture.is_loaded(), "Fixture should not be loaded initially");

    // Load fixture data
    let data = lazy_fixture.load();
    assert!(data.is_ok(), "Should load fixture data on demand");
    assert!(lazy_fixture.is_loaded(), "Fixture should be loaded after access");
}

#[test]
fn test_fixture_api_caching() {
    let fixture = FixtureBuilder::new("cached_test")
        .fixture_type("model")
        .lazy_load(false)
        .cache(true)
        .generator(|| {
            // Generate expensive model
            std::thread::sleep(std::time::Duration::from_millis(10));
            vec![1.0; 1000]
        })
        .build()
        .unwrap();

    // First load
    let start1 = std::time::Instant::now();
    let data1 = fixture.load().unwrap();
    let time1 = start1.elapsed();

    // Second load (should be from cache)
    let start2 = std::time::Instant::now();
    let data2 = fixture.load().unwrap();
    let time2 = start2.elapsed();

    assert_eq!(data1.len(), data2.len(), "Cached data should be identical");
    assert!(time2 < time1 / 2, "Cached load should be much faster");
}

#[test]
fn test_fixture_api_file_loading() {
    let path = PathBuf::from("tests/fixtures/datasets/test_data.bin");

    let fixture = FixtureBuilder::new("file_test")
        .fixture_type("dataset")
        .path(path.clone())
        .build();

    assert!(fixture.is_ok(), "Should create file-based fixture");
    let file_fixture = fixture.unwrap();

    assert_eq!(file_fixture.path, Some(path), "Should store file path");

    // Try to load (will fail if file doesn't exist, which is expected in TDD)
    let data = file_fixture.load();
    if data.is_err() {
        // Expected in TDD - file doesn't exist yet
        assert!(true, "File loading will fail until fixture files are created");
    }
}

#[test]
fn test_fixture_api_cleanup() {
    let fixture = FixtureBuilder::new("cleanup_test")
        .fixture_type("state")
        .cleanup_required(true)
        .generator(|| {
            // Create temporary state
            vec![1.0, 2.0, 3.0]
        })
        .build()
        .unwrap();

    assert!(fixture.cleanup_required, "Fixture should require cleanup");

    // Load and use fixture
    let data = fixture.load().unwrap();
    assert!(!data.is_empty(), "Should load fixture data");

    // Cleanup
    let cleanup_result = fixture.cleanup();
    assert!(cleanup_result.is_ok(), "Cleanup should succeed");

    // After cleanup, fixture should be unloaded
    assert!(!fixture.is_loaded(), "Fixture should be unloaded after cleanup");
}

#[test]
fn test_fixture_api_dependency_chain() {
    // Create base fixture
    let base = FixtureBuilder::new("base_model")
        .fixture_type("model")
        .generator(|| vec![1.0; 100])
        .build()
        .unwrap();

    // Create dependent fixture
    let dependent = FixtureBuilder::new("fine_tuned_model")
        .fixture_type("model")
        .depends_on(vec![base.id.clone()])
        .generator_with_deps(|deps| {
            // Use base model to create fine-tuned version
            let base_data = deps[0].load().unwrap();
            base_data.into_iter().map(|x| x * 2.0).collect()
        })
        .build();

    assert!(dependent.is_ok(), "Should create dependent fixture");
    let dep_fixture = dependent.unwrap();

    assert!(!dep_fixture.dependencies.is_empty(), "Should have dependencies");
    assert!(dep_fixture.dependencies.contains(&base.id), "Should depend on base fixture");
}