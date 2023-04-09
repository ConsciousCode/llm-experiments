from pymilvus import connections, DataType, CollectionSchema, FieldSchema, Collection
import torch

class ExternalMemory(torch.autograd.Function):
	'''
	Discretized external memory query-and-update operation. The external memory
	should approximate attention over an arbitrarily large external memory vector.
	
	1. Query a number of entries from the memory
	2. Use the gate to select keys and values and the add/erase operation
	3. Update the memory with the new keys and values
	4. Return the original query
	'''
	
	@staticmethod
	def forward(ctx, q, k, v, memory):
		'''
		Parameters:
			q: Query vector
			k: Key vector
			v: Value vector
			g: Gate vector, usually a sigmoid
			mem: Memory interface
			remember: Number of entries to remember per query
		
		If remember/retain/forget are 0, query/add/remove operations are skipped.
		'''
		print("qkv", q.shape, k.shape, v.shape)
		
		mk, mv = memory.search(q)
		memory.add(k, v)
		return mk, mv

	@staticmethod
	def backward(ctx, mk_grad, mv_grad):
		'''
		Parameters:
			mk_grad: Gradient of memory keys
			mv_grad: Gradient of memory values
		
		This uses STE to approximate the gradient of the memory.
		'''
		
		return mk_grad, mk_grad, mv_grad, None

from vespa.application import ApplicationPackage
from vespa.package import Field, Document, Schema
import torch

class KNNMemory:
	'''
	kNN memory interfacing with the Vespa database.
	'''

	def __init__(self, name, kdim, vdim):
		self.name = name
		self.kdim = kdim
		self.vdim = vdim

		# Define the schema
		self.app = ApplicationPackage(name, [
			Schema("infostill",
				Document([
					Field("id", "int", indexing=["attribute"], attribute=["fast-search"]),
					Field("key", f"tensor<float>(x[{kdim}])", indexing=["attribute"], attribute=["fast-search"]),
					Field("value", f"tensor<float>(x[{vdim}])", indexing=["attribute"])
				])
			)
		])

	def add(self, keys, values):
		assert keys.shape[:-1] == values.shape[:-1], "Keys and values should have matching batch and sequence dimensions"
		
		keys = keys.reshape(-1, keys.shape[-1])
		vals = values.reshape(-1, values.shape[-1])

		for key, val in zip(keys, vals):
			self.app.feed_data_point(
				schema="infostill",
				data_id=int(self.app.get_data_count("infostill")),
				fields={
					"key": key.tolist(),
					"value": val.tolist()
				}
			)

	def search(self, queries, top_k=1):
		print("Search", queries.shape, queries)
		batch, seq, dim = queries.shape
		queries = queries.reshape(-1, dim)
		
		keys = torch.empty((batch, seq * top_k, dim), dtype=torch.float32)
		values = torch.empty((batch, seq * top_k, dim), dtype=torch.float32)

		for i, query in enumerate(queries):
			ann_query = {
				"yql": "select * from sources * where ([{'targetHits':%(top_k)s}]nearestNeighbor(key, key_embedding));",
				"ranking.profile": "my_profile",
				"ranking.features.query(key_embedding)": query.tolist(),
				"hits": top_k
			}

			result = self.app.query(body=ann_query)
			row = i // seq
			col = (i % seq) * top_k * dim

			for hit in result.hits:
				keys[row, col:col + dim] = torch.tensor(hit["fields"]["key"], dtype=torch.float32)
				values[row, col:col + dim] = torch.tensor(hit["fields"]["value"], dtype=torch.float32)
				col += dim

		return keys, values
