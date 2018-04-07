class TopDownParser(object):
    def __init__(self, model, env, dataset):
        self._featurizer = model._featurizer
        self._policy = model._policy
        self._policy_act_prob = model._policy_act_prob
        self._policy_act_pos_prob = model._policy_act_pos_prob
        self._desc_prob = model._desc_prob
        self._describer = model._describer
        self._segmenter = model._segmenter

        self._env = env
        self._dataset = dataset

    @profile
    def parse(self, seq_batch):
        demos = [task.demonstration() for task in seq_batch.tasks]

        state_feature_cache = {}
        world_feature_cache = {}
        # TODO dup dup dup
        def cache(d, start=0, clear=False):
            if clear:
                state_feature_cache.clear()
                world_feature_cache.clear()

            demo = demos[d]
            init_state_obs = (seq_batch.all_state_obs[d, start, ...]
                .unsqueeze(0).expand((len(demo), self._env.n_state_obs)))
            init_world_obs = (seq_batch.all_world_obs[d, start, ...]
                .unsqueeze(0).expand((len(demo), self._env.n_world_obs) + self._env.world_shape))
            mid_state_obs = torch.stack([seq_batch.all_state_obs[d, i, ...] for i in range(len(demo))])
            mid_world_obs = torch.stack([seq_batch.all_world_obs[d, i, ...] for i in range(len(demo))])
            end_state_obs = (seq_batch.last_state_obs[d, ...]
                .unsqueeze(0).expand((len(demo), self._env.n_state_obs)))
            end_world_obs = (seq_batch.last_world_obs[d, ...]
                .unsqueeze(0).expand((len(demo), self._env.n_world_obs) + self._env.world_shape))
            state_feats1, world_feats1 = self._featurizer(
                init_state_obs, mid_state_obs, init_world_obs, mid_world_obs)
            state_feats2, world_feats2 = self._featurizer(
                mid_state_obs, end_state_obs, mid_world_obs, end_world_obs)
            #for i in range(1, len(demo)-1):
            for i in range(start, len(demo)-1):
                #print((start, i), (i, len(demo)-1))
                state_feature_cache[d, start, i] = state_feats1[i, :]
                state_feature_cache[d, i, len(demo)-1] = state_feats2[i, :]
                world_feature_cache[d, start, i] = world_feats1[i, ...]
                world_feature_cache[d, i, len(demo)-1] = world_feats2[i, ...]

        def get_feats(d, i, j):
            return state_feature_cache[d, i, j], world_feature_cache[d, i, j]

        # TODO bad score_parent
        def score_span(d, pi, pdesc, i, j, desc, score_parent=True):
            ### TODO util function
            ##demo = demos[d]
            ##hot_desc = np.zeros((max(len(desc), len(pdesc)), j-i+1+1, len(self._dataset.vocab)))
            ###hot_desc = np.zeros((max(len(desc), len(pdesc)), 1+1, len(self._dataset.vocab)))
            ##for p, t in enumerate(pdesc):
            ##    hot_desc[p, 0, t] = 1
            ##for p, t in enumerate(desc):
            ##    hot_desc[p, 1:, t] = 1

            ##top_obs = [seq_batch.all_obs[d, i, :]]
            ##top_init_obs = [seq_batch.all_obs[d, pi, :]]
            ##top_actions = [self._env.SAY]
            ##top_actions_pos = [0]
            ##top_actions_pos_mask = [0]
            ##top_desc_out_mask = [1]
            ##top_desc = [desc]

            ##obs = [seq_batch.all_obs[d, k, :] for k in range(i, j+1)]
            ##init_obs = [seq_batch.all_obs[d, i, :] for _ in range(i, j+1)]
            ##actions = [demo[k][1][0] for k in range(i, j)] + [self._env.STOP]
            ### TODO not here
            ##actions_pos = [demo[k][1][1] for k in range(i, j)] + [None]
            ##actions_pos = [None if ap is None else int(np.ravel_multi_index(ap,
            ##    seq_batch.init_obs.shape[2:])) for ap in actions_pos]
            ##actions_pos_mask = [0 if ap is None else 1 for ap in actions_pos]
            ##actions_pos = [0 if ap is None else ap for ap in actions_pos]
            ##desc_out_mask = [0 for _ in actions]
            ##desc = [[None] for _ in actions]
            ###obs = [seq_batch.all_obs[d, j, :]]
            ###init_obs = [seq_batch.all_obs[d, i, :]]
            ###actions = [self._env.STOP]
            ###actions_pos = [0]
            ###actions_pos_mask = [0]
            ###desc_out_mask = [0]
            ###desc = [[None]]

            ##if score_parent:
            ##    obs = top_obs + obs
            ##    init_obs = top_init_obs + init_obs
            ##    actions = top_actions + actions
            ##    actions_pos = top_actions_pos + actions_pos
            ##    actions_pos_mask = top_actions_pos_mask + actions_pos_mask
            ##    desc_out_mask = top_desc_out_mask + desc_out_mask
            ##    desc = top_desc + desc
            ##else:
            ##    hot_desc = hot_desc[:, 1:, ...]

            ##obs = torch.stack(obs)
            ##init_obs = torch.stack(init_obs)
            ##desc_out, desc_out_target = data.load_desc_data(
            ##    desc, self._dataset, target=True, tokenize=False)

            ### TODO Batch.of_x
            ##step_batch = data.StepBatch(
            ##    init_obs,
            ##    obs,
            ##    Variable(torch.LongTensor(actions)),
            ##    Variable(torch.LongTensor(actions_pos)),
            ##    Variable(torch.FloatTensor(actions_pos_mask)),
            ##    None,
            ##    Variable(torch.FloatTensor(hot_desc)),
            ##    Variable(torch.FloatTensor(desc_out_mask)),
            ##    Variable(torch.FloatTensor(desc_out)),
            ##    Variable(torch.LongTensor(desc_out_target)))
            ##if next(self._policy.parameters()).is_cuda:
            ##    step_batch = step_batch.cuda()

            ##feats, conv_feats = self._featurizer(step_batch.init_obs, step_batch.obs)
            ##act_logits, _, (_, desc_logits) = self._policy(feats, conv_feats, step_batch)
            ##scores = self._policy_prob(act_logits, step_batch.act)
            ##return scores

            assert not score_parent
            demo = demos[d]

            hot_desc = np.zeros((len(desc), j-i+1, len(self._dataset.vocab)))
            for p, t in enumerate(desc):
                hot_desc[p, :, t] = 1

            state_feats = []
            world_feats = []
            actions = []
            actions_pos = []
            actions_pos_mask = []
            for k in range(i, j):
                sf, wf = get_feats(d, i, k)
                state_feats.append(sf)
                world_feats.append(wf)
                a, ap = demo[k][1]
                actions.append(a)
                actions_pos.append(
                    0 if ap is None else int(np.ravel_multi_index(ap, wf.shape[1:])))
                actions_pos_mask.append(0 if ap is None else 1)
            sf, wf = get_feats(d, i, j)
            state_feats.append(sf)
            world_feats.append(wf)
            actions.append(self._env.STOP)
            actions_pos.append(0)
            actions_pos_mask.append(0)

            state_feats = torch.stack(state_feats)
            world_feats = torch.stack(world_feats)
            step_batch = data.StepBatch(
                None, None, None, None,
                Variable(torch.LongTensor(actions)),
                Variable(torch.LongTensor(actions_pos)),
                Variable(torch.FloatTensor(actions_pos_mask)),
                None,
                Variable(torch.FloatTensor(hot_desc)),
                None, None, None)
            if next(self._policy.parameters()).is_cuda:
                step_batch = step_batch.cuda()

            act_logits, act_pos_logits, _ = self._policy(state_feats, world_feats, step_batch)
            scores = (
                self._policy_act_prob(act_logits, step_batch.act)
                + (self._policy_act_pos_prob(act_pos_logits, step_batch.act_pos)
                    * step_batch.act_pos_mask))
            return scores

        def propose_descs(indices, n):
            #for index in indices:
            #    print(get_feats(*index).sum().data.cpu().numpy())
            reps = torch.cat(
                [get_feats(*index)[0].unsqueeze(0).expand((n, -1)) 
                    for index in indices],
                0)
            # TODO magic
            descs = self._describer.decode(reps.unsqueeze(0), 20)
            return descs

        def propose_splits(d, n):
            demo = demos[d]
            # TODO expand
            first_reps = torch.stack(
                [get_feats(d, 0, i)[0] for i in range(1, len(demo)-1)])
            second_reps = torch.stack(
                [get_feats(d, i, len(demo)-1)[0] for i in range(1, len(demo)-1)])
            splits = self._segmenter(first_reps)  + self._segmenter(second_reps)
            _, top = splits.topk(min(n, splits.shape[0]), sorted=False)
            return 1 + top

        def render(desc):
            return ' '.join(self._dataset.vocab.get(t) for t in desc[1:-1])

        # 0   1   2   3   4   5   6   7   8   9
        # s a s a s a s a s a s a s a s a s a s STOP
        #
        # (0, 5)
        # final obs is s5
        # replace a5 with STOP
        # 
        # (5, 9)
        # final obs is s9
        # replace a9 with STOP

        out_descs = []
        out_actions = []
        out_states = []

        for d, demo in enumerate(demos):
            if len(demo) < 3:
                continue
            cache(d, clear=True)
            # TODO magic
            splits = unwrap(propose_splits(d, 5))
            #splits = [1]
            # TODO gross
            pdesc = ling.tokenize(seq_batch.tasks[d].desc, self._dataset.vocab)

            parent_scores = score_span(d, 0, pdesc, 0, len(demo)-1, pdesc, score_parent=False)

            indices = [[(d, 0, k), (d, k, len(demo)-1)] for k in splits]
            indices = sum(indices, [])
            descs = propose_descs(indices, 1)
            desc_pairs = [(descs[2*i], descs[2*i+1]) for i in range(len(splits))]

            pick_scores = []
            pick_splits = []
            pick_descs = []
            assert len(splits) == len(desc_pairs)
            for k, (desc1, desc2) in zip(splits, desc_pairs):
                cache(d, 0, clear=True)
                cache(d, k)
                if 0 in desc1 or 0 in desc2:
                    continue
                #print(render(pdesc))
                #print(render(propose_descs([(d, 0, len(demo)-1)], 1)[0]))
                #print(render(desc1), render(desc2))
                #return

                s1c, = unwrap(score_span(d, 0, pdesc, 0, k, desc1, score_parent=False).sum())
                s2c, = unwrap(score_span(d, 0, pdesc, k, len(demo)-1, desc2, score_parent=False).sum())

                s1p, = unwrap(parent_scores[:k].sum())
                s2p, = unwrap(parent_scores[k:].sum())

                pick_desc = [None, None]

                if s1c < s1p:
                    s1 = s1c
                    pick_desc[0] = desc1
                else:
                    s1 = s1p

                if s2c < s2p:
                    s2 = s2c
                    pick_desc[1] = desc2
                else:
                    s2 = s2p

                pick_scores.append(s1 + s2)
                pick_splits.append(k)
                pick_descs.append(tuple(pick_desc))
                #score_parts.append((s1, s2, s1a, s2a, s2b))

            if len(pick_scores) == 0:
                continue
            # TODO argmin on gpu?
            i_split = np.asarray(pick_scores).argmin()
            split = pick_splits[i_split]
            score = pick_scores[i_split]
            d1, d2 = pick_descs[i_split]
            actions = [t[1][0] for t in demo]

            if not (d1 is None and d2 is None) and np.random.random() < 0.05:
                print(
                    render(pdesc), 
                    ':', 
                    render(d1) if d1 else '_', 
                    '>', 
                    render(d2) if d2 else '_')
                print(actions[:split], actions[split:-1])
                print()

            ### # TODO HORRIBLE
            ### if d1 is not None:
            ###     out_descs.append(render(d1))
            ###     out_actions.append(actions[0:split] + [self._env.STOP])
            ###     out_states.append(seq_batch.all_obs[d, 0:split+1, :].data)

            ### if d2 is not None:
            ###     out_descs.append(render(d2))
            ###     assert actions[len(demo)-1][0] == self._env.STOP
            ###     out_actions.append(actions[split:len(demo)-1] + [self._env.STOP])
            ###     out_states.append(seq_batch.all_obs[d, split:len(demo), :].data)

        #return data.ParseBatch(out_descs, out_actions, out_states)
        return None

            #print(seq_batch.tasks[d].desc)
            #print(splits)
            #print(actions[:split], actions[split:])
            #print(render(d1), render(d2))

