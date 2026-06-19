#!/usr/bin/env bash
#
# generate_proto.sh
#
# Compiles all .proto files in proto/ into Python modules using grpc_tools.
# Output goes to proto/generated/. Generated files include:
#   - *_pb2.py        : message classes (serialization/deserialization)
#   - *_pb2_grpc.py   : gRPC client stubs and server base classes
#
# The script also fixes imports in the generated gRPC files. By default,
# grpc_tools emits bare imports (e.g., "import nodes_pb2 as ...") which
# break when the files live inside a Python package. This script rewrites
# them to relative imports (e.g., "from . import nodes_pb2 as ...").
#
# Usage:
#   bash scripts/generate_proto.sh
#   make proto
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$REPO_ROOT/proto"
OUT_DIR="$REPO_ROOT/proto/generated"

mkdir -p "$OUT_DIR"

python3 -m grpc_tools.protoc -I "$PROTO_DIR" --python_out="$OUT_DIR" --grpc_python_out="$OUT_DIR" "$PROTO_DIR"/*.proto

touch "$OUT_DIR/__init__.py"

for f in "$OUT_DIR"/*_pb2_grpc.py; do
    [ -f "$f" ] || continue
    sed -i 's/^import \(.*_pb2\) as/from . import \1 as/' "$f"
done

echo "Proto generation complete. Output: $OUT_DIR"
