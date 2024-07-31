use super::{Metadata, MetadataValueConversionError, SegmentScope, SegmentScopeConversionError};
use crate::{
    chroma_proto,
    errors::{ChromaError, ErrorCodes},
};
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SegmentType {
    HnswDistributed,
    BlockfileMetadata,
    BlockfileRecord,
    Sqlite,
}

impl From<SegmentType> for String {
    fn from(segment_type: SegmentType) -> String {
        match segment_type {
            SegmentType::HnswDistributed => {
                "urn:chroma:segment/vector/hnsw-distributed".to_string()
            }
            SegmentType::BlockfileRecord => "urn:chroma:segment/record/blockfile".to_string(),
            SegmentType::Sqlite => "urn:chroma:segment/metadata/sqlite".to_string(),
            SegmentType::BlockfileMetadata => "urn:chroma:segment/metadata/blockfile".to_string(),
        }
    }
}

impl TryFrom<&str> for SegmentType {
    type Error = SegmentConversionError;

    fn try_from(segment_type: &str) -> Result<Self, Self::Error> {
        match segment_type {
            "urn:chroma:segment/vector/hnsw-distributed" => Ok(SegmentType::HnswDistributed),
            "urn:chroma:segment/record/blockfile" => Ok(SegmentType::BlockfileRecord),
            "urn:chroma:segment/metadata/sqlite" => Ok(SegmentType::Sqlite),
            "urn:chroma:segment/metadata/blockfile" => Ok(SegmentType::BlockfileMetadata),
            _ => Err(SegmentConversionError::InvalidSegmentType),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Segment {
    pub(crate) id: Uuid,
    pub(crate) r#type: SegmentType,
    pub(crate) scope: SegmentScope,
    pub(crate) collection: Option<Uuid>,
    pub(crate) metadata: Option<Metadata>,
    pub(crate) file_path: HashMap<String, Vec<String>>,
    // Configuration is currently transported as json, in the future
    // we should have a more structured way to transport and represent
    // configuration
    // This was an explicit shortcut to avoid having to define a new
    // proto message for configuration and per segment type configuration
    // https://github.com/chroma-core/chroma/issues/2598
    pub(crate) configuration_json: Option<Value>,
}

#[derive(Error, Debug)]
pub(crate) enum SegmentConversionError {
    #[error("Invalid UUID")]
    InvalidUuid,
    #[error(transparent)]
    MetadataValueConversionError(#[from] MetadataValueConversionError),
    #[error(transparent)]
    SegmentScopeConversionError(#[from] SegmentScopeConversionError),
    #[error("Invalid segment type")]
    InvalidSegmentType,
    #[error(transparent)]
    SerdeJsonError(#[from] serde_json::Error),
}

impl ChromaError for SegmentConversionError {
    fn code(&self) -> crate::errors::ErrorCodes {
        match self {
            SegmentConversionError::InvalidUuid => ErrorCodes::InvalidArgument,
            SegmentConversionError::InvalidSegmentType => ErrorCodes::InvalidArgument,
            SegmentConversionError::SegmentScopeConversionError(e) => e.code(),
            SegmentConversionError::MetadataValueConversionError(e) => e.code(),
            SegmentConversionError::SerdeJsonError(_) => ErrorCodes::InvalidArgument,
        }
    }
}

impl TryFrom<chroma_proto::Segment> for Segment {
    type Error = SegmentConversionError;

    fn try_from(proto_segment: chroma_proto::Segment) -> Result<Self, Self::Error> {
        let mut proto_segment = proto_segment;

        let segment_uuid = match Uuid::try_parse(&proto_segment.id) {
            Ok(uuid) => uuid,
            Err(_) => return Err(SegmentConversionError::InvalidUuid),
        };
        let collection_uuid = match proto_segment.collection {
            Some(collection_id) => match Uuid::try_parse(&collection_id) {
                Ok(uuid) => Some(uuid),
                Err(_) => return Err(SegmentConversionError::InvalidUuid),
            },
            // The UUID can be none in the local version of chroma but not distributed
            None => return Err(SegmentConversionError::InvalidUuid),
        };
        let segment_metadata: Option<Metadata> = match proto_segment.metadata {
            Some(proto_metadata) => match proto_metadata.try_into() {
                Ok(metadata) => Some(metadata),
                Err(e) => return Err(SegmentConversionError::MetadataValueConversionError(e)),
            },
            None => None,
        };
        let scope: SegmentScope = match proto_segment.scope.try_into() {
            Ok(scope) => scope,
            Err(e) => return Err(SegmentConversionError::SegmentScopeConversionError(e)),
        };

        let segment_type = match proto_segment.r#type.as_str() {
            "urn:chroma:segment/vector/hnsw-distributed" => SegmentType::HnswDistributed,
            "urn:chroma:segment/record/blockfile" => SegmentType::BlockfileRecord,
            "urn:chroma:segment/metadata/sqlite" => SegmentType::Sqlite,
            "urn:chroma:segment/metadata/blockfile" => SegmentType::BlockfileMetadata,
            _ => {
                return Err(SegmentConversionError::InvalidSegmentType);
            }
        };

        let mut file_paths = HashMap::new();
        let drain = proto_segment.file_paths.drain();
        for (key, value) in drain {
            file_paths.insert(key, value.paths);
        }

        let configuration_json = match proto_segment.configuration_json_str {
            Some(json_str) => match serde_json::from_str(&json_str) {
                Ok(json) => Some(json),
                Err(e) => {
                    return Err(SegmentConversionError::SerdeJsonError(e));
                }
            },
            None => None,
        };

        println!("HAMMAD CONFIGURATION JSON: {:?}", configuration_json);

        Ok(Segment {
            id: segment_uuid,
            r#type: segment_type,
            scope: scope,
            collection: collection_uuid,
            metadata: segment_metadata,
            file_path: file_paths,
            configuration_json,
        })
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::types::MetadataValue;

    #[test]
    fn test_segment_try_from() {
        let mut metadata = chroma_proto::UpdateMetadata {
            metadata: HashMap::new(),
        };
        metadata.metadata.insert(
            "foo".to_string(),
            chroma_proto::UpdateMetadataValue {
                value: Some(chroma_proto::update_metadata_value::Value::IntValue(42)),
            },
        );

        let configuration_json = r#"{"M": 16, "ef_construction": 200, "ef_search": 200}"#;

        let proto_segment = chroma_proto::Segment {
            id: "00000000-0000-0000-0000-000000000000".to_string(),
            r#type: "urn:chroma:segment/vector/hnsw-distributed".to_string(),
            scope: chroma_proto::SegmentScope::Vector as i32,
            collection: Some("00000000-0000-0000-0000-000000000000".to_string()),
            metadata: Some(metadata),
            file_paths: HashMap::new(),
            configuration_json_str: Some(configuration_json.to_string()),
        };
        let converted_segment: Segment = proto_segment.try_into().unwrap();
        assert_eq!(converted_segment.id, Uuid::nil());
        assert_eq!(converted_segment.r#type, SegmentType::HnswDistributed);
        assert_eq!(converted_segment.scope, SegmentScope::VECTOR);
        assert_eq!(converted_segment.collection, Some(Uuid::nil()));
        let metadata = converted_segment.metadata.unwrap();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata.get("foo").unwrap(), &MetadataValue::Int(42));
        assert_eq!(
            converted_segment.configuration_json.unwrap(),
            serde_json::from_str::<serde_json::Value>(configuration_json).unwrap(),
        );
    }
}
