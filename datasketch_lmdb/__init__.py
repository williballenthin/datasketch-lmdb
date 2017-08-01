import logging

import lmdb
import msgpack
import datasketch
import datasketch.lsh


logger = logging.getLogger(__name__)


class LMDBMinHashLSH(datasketch.MinHashLSH):
    '''
    MinHash LSH indexed backed by LMDB.
    
    You must close this object once you're done with it.
    See :ref:`LMDBMinHashLSH.close`.
    For example::
    
        lsh = LMDBMinHashLSH(path='lsh.bin', threshold=0.5, num_perm=128)
        ...
        lsh.close()

    Or more safely::

        import contextlib
        with contextlib.closing(LMDBMinHashLSH(path='lsh.bin', threshold=0.5, num_perm=128)) as lsh:
            ...
            lsh.insert("m2", m2)
    
    Note, you may not pickle this type of LSH, since it maintains a handle to a persistent database.

    see: :ref:`datasketch.lsh.MinHashLSH`.
    '''
    def __init__(self, path, threshold=0.9, num_perm=128, weights=(0.5,0.5), params=None):
        super(LMDBMinHashLSH, self).__init__(threshold=threshold, num_perm=num_perm, weights=weights, params=params)

        # cleanup members from parent class.
        del self.hashtables
        del self.keys

        # no need to persist.
        # immutable.
        # type: List[Tuple[int, int]]
        self.hashranges  # expected nop, constructed by parent.

        db_count = self.b + 1  # b * hashtable_dbs + 1 * key_db
        self.env = lmdb.open(path, max_dbs=db_count, writemap=True)

        # database 0.
        # mutable.
        # type: Mapping[Key, List[bytes]]
        self.keys_db = self.env.open_db('keys'.encode('ascii'))

        # database 1 through database (1 + self.b)
        # mutable.
        # type: List[Mapping[bytes, List[Key]]]
        self.hashtable_dbs = [self.env.open_db(('hashtable_%d' % (i)).encode('ascii')) for i in range(self.b)]

    def close(self):
        '''
        Close the handle to the underlying LMDB database.
        This instance is not longer valid after calling `.close`.
 
        You *must* close this object once you're done with it.
        For example::

            lsh = LMDBMinHashLSH(path='lsh.bin', threshold=0.5, num_perm=128)
            ...
            lsh.close()

        Or more safely::

            import contextlib
            with contextlib.closing(LMDBMinHashLSH(path='lsh.bin', threshold=0.5, num_perm=128)) as lsh:
                ...
                lsh.insert("m2", m2)
        '''
        self.env.close()

    # override
    def insert(self, key, minhash):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))

        bkey = msgpack.packb(key)
        with self.env.begin(write=True, buffers=True) as txn:
            bhashes = txn.get(bkey, db=self.keys_db)
            if bhashes is not None:
                raise ValueError("The given key already exists")

            # keys[key] = [self._H(minhash.hashvalues[start:end]) for start, end in self.hashranges]
            hashes = [self._H(minhash.hashvalues[start:end]) for start, end in self.hashranges]
            bhashes = msgpack.packb(hashes)
            txn.put(bkey, bhashes, overwrite=True, db=self.keys_db)

            for i, (H, hashtable_db) in enumerate(zip(hashes, self.hashtable_dbs)):
                # hashtable[H].append(key)
                bkeys = txn.get(H, db=hashtable_db)
                if bkeys is not None:
                    keys = msgpack.unpackb(bkeys)
                else:
                    keys = []
                keys.append(key)
                bkeys = msgpack.packb(keys)
                txn.put(H, bkeys, overwrite=True, db=hashtable_db)

    # override
    def query(self, minhash):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))

        candidates = set()
        with self.env.begin(write=False, buffers=True) as txn:
            for i, ((start, end), hashtable_db) in enumerate(zip(self.hashranges, self.hashtable_dbs)):
                H = self._H(minhash.hashvalues[start:end])
                bkeys = txn.get(H, db=hashtable_db)
                if bkeys is not None:
                    keys = msgpack.unpackb(bkeys, use_list=False)
                    for key in keys: 
                        candidates.add(key)

            return candidates

    # override
    def __contains__(self, key):
        bkey = msgpack.packb(key)
        with self.env.begin(write=False, buffers=True) as txn:
            bhashes = txn.get(bkey, db=self.keys_db)
            return bhashes is not None

    # override
    def remove(self, key):
        bkey = msgpack.packb(key)
        with self.env.begin(write=True, buffers=True) as txn:
            bhashes = txn.get(bkey, db=self.keys_db)
            if bhashes is None:
                raise ValueError("The given key does not exist")
            hashes = msgpack.unpackb(bhashes)

            for i, (H, hashtable_db) in enumerate(zip(hashes, self.hashtable_dbs)):
                # hashtable[H].remove(key)
                keys = txn.get(H, db=hashtable_db)
                if keys is not None:
                    keys = msgpack.unpackb(keys)
                    try:
                        keys.remove(key)
                    except ValueError:
                        # probably shouldn't actually get here?.
                        pass
                    else:
                        # key wasn't there, no need to update the value
                        if keys:
                            keys = msgpack.packb(keys)
                            txn.put(H, keys, overwrite=True, db=hashtable_db)
                        else:
                            # there's nothing left in the list, so delete it.
                            #
                            # if not hashtable[H]:
                            #     del hashtable[H]
                            txn.delete(H, db=hashtable_db)

            # self.keys.pop(key)    
            txn.pop(bkey, db=self.keys_db)

    # override
    def is_empty(self):
        with self.env.begin(write=False, buffers=True) as txn:
            for hashtable_db in self.hashtable_dbs:
                if txn.stat(hashtable_db)['entries'] > 0:
                    return False
        return True

    # override
    def _query_b(self, minhash, b):
        raise NotImplementedError()


if __name__ == '__main__':
    import shutil
    import contextlib
    logging.basicConfig(level=logging.DEBUG)

    set1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
                'estimating', 'the', 'similarity', 'between', 'datasets'])
    set2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
                'estimating', 'the', 'similarity', 'between', 'documents'])
    set3 = set(['minhash', 'is', 'probability', 'data', 'structure', 'for',
                'estimating', 'the', 'similarity', 'between', 'documents'])

    m1 = datasketch.MinHash(num_perm=128)
    m2 = datasketch.MinHash(num_perm=128)
    m3 = datasketch.MinHash(num_perm=128)
    for d in set1:
        m1.update(d.encode('utf8'))
    for d in set2:
        m2.update(d.encode('utf8'))
    for d in set3:
        m3.update(d.encode('utf8'))

    minhashes = {
        'm1': m1,
        'm2': m2,
        'm3': m3,
    }

    with contextlib.closing(LMDBMinHashLSH(path='lsh.bin', threshold=0.5, num_perm=128)) as lsh:
        lsh.insert("m2", m2)
        lsh.insert("m3", m3)
        results = lsh.query(m1)
        print("Approximate neighbours with Jaccard similarity > 0.5", results)
        for result in results:
            print('  - %s: %0.2f' % (result, m1.jaccard(minhashes[result])))
 
        lsh.remove("m3")
        results = lsh.query(m1)
        print("Approximate neighbours with Jaccard similarity > 0.5", results)
        for result in results:
            print('  - %s: %0.2f' % (result, m1.jaccard(minhashes[result])))
    shutil.rmtree('lsh.bin')

    with contextlib.closing(LMDBMinHashLSH(path='lsh.bin', threshold=0.7, num_perm=128)) as lsh:
        lsh.insert("m2", m2)
        lsh.insert("m3", m3)
        results = lsh.query(m1)
        print("Approximate neighbours with Jaccard similarity > 0.7", results)
        for result in results:
            print('  - %s: %0.2f' % (result, m1.jaccard(minhashes[result])))
    shutil.rmtree('lsh.bin')
