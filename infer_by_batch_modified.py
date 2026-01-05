import os
import json
import clip
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from ClipCap import ClipCaptionModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from utils import compose_discrete_prompts
from load_annotations import load_entities_text
from search import greedy_search, beam_search, opt_search
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories

# MeaCap模块导入（可选）
try:
    from utils.detect_utils import retrieve_concepts
    from models.clip_utils import CLIP
    MEA_CAP_AVAILABLE = True
except ImportError:
    MEA_CAP_AVAILABLE = False
    print("Warning: MeaCap modules not found. Only ViECap original method available.")

@torch.no_grad()
def main(args) -> None:
    # initializing
    device = args.device
    clip_name = args.clip_model.replace('/', '') 
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512

    # ========== MeaCap 记忆库初始化（如果使用） ==========
    if args.use_memory and MEA_CAP_AVAILABLE:
        print("Initializing MeaCap memory bank...")
        # 1. 加载SceneGraphParser
        parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint)
        parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint)
        parser_model.eval()
        parser_model.to(device)
        print(f'SceneGraphParser loaded from {args.parser_checkpoint}')
        
        # 2. 加载SentenceBERT
        wte_model = SentenceTransformer(args.wte_model_path)
        print(f'SentenceBERT loaded from {args.wte_model_path}')
        
        # 3. 加载CLIP（用于记忆库检索）
        vl_model = CLIP(args.vl_model)
        vl_model.device_convert(device)
        print(f'CLIP model loaded from {args.vl_model}')
        
        # 4. 加载记忆库
        memory_id = args.memory_id
        memory_caption_path = os.path.join(f"data/memory/{memory_id}", "memory_captions.json")
        memory_clip_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_clip_embeddings.pt")
        memory_wte_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_wte_embeddings.pt")
        
        if not os.path.exists(memory_caption_path):
            raise FileNotFoundError(f"Memory caption file not found: {memory_caption_path}")
        if not os.path.exists(memory_clip_embedding_file):
            raise FileNotFoundError(f"Memory CLIP embedding file not found: {memory_clip_embedding_file}")
        
        memory_clip_embeddings = torch.load(memory_clip_embedding_file, map_location='cpu')
        memory_wte_embeddings = torch.load(memory_wte_embedding_file, map_location='cpu') if os.path.exists(memory_wte_embedding_file) else None
        with open(memory_caption_path, 'r') as f:
            memory_captions = json.load(f)
        
        print(f'Memory bank loaded: {memory_id}')
        print(f'Memory size: {len(memory_captions)} captions')
        
        # 大记忆库可能需要CPU检索
        if memory_id in ['cc3m', 'ss1m']:
            retrieve_on_CPU = True
            print('Large memory bank detected, will retrieve on CPU')
        else:
            retrieve_on_CPU = False
    else:
        parser_model = None
        parser_tokenizer = None
        wte_model = None
        vl_model = None
        memory_clip_embeddings = None
        memory_wte_embeddings = None
        memory_captions = None
        retrieve_on_CPU = False
        if args.use_memory and not MEA_CAP_AVAILABLE:
            print("Warning: --use_memory specified but MeaCap modules not available. Using ViECap original method.")
            args.use_memory = False

    # ========== ViECap 原始实体词汇表加载（如果不使用记忆库） ==========
    if not args.use_memory:
        # loading categories vocabulary for objects
        if args.name_of_entities_text == 'visual_genome_entities':
            entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/all_objects_attributes_relationships.pickle', not args.disable_all_entities)
            if args.prompt_ensemble: # loading ensemble embeddings
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/visual_genome_embedding_{clip_name}_with_ensemble.pickle')
            else:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/visual_genome_embedding_{clip_name}.pickle')
        elif args.name_of_entities_text == 'coco_entities':
            entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/coco_categories.json', not args.disable_all_entities)
            if args.prompt_ensemble:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/coco_embeddings_{clip_name}_with_ensemble.pickle')
            else:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/coco_embeddings_{clip_name}.pickle')
        elif args.name_of_entities_text == 'open_image_entities':
            entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/oidv7-class-descriptions-boxable.csv', not args.disable_all_entities)
            if args.prompt_ensemble:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/open_image_embeddings_{clip_name}_with_ensemble.pickle')
            else:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/open_image_embeddings_{clip_name}.pickle')
        elif args.name_of_entities_text == 'vinvl_vg_entities':
            entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/VG-SGG-dicts-vgoi6-clipped.json', not args.disable_all_entities)
            if args.prompt_ensemble:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vg_embeddings_{clip_name}_with_ensemble.pickle')
            else:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vg_embeddings_{clip_name}.pickle')
        elif args.name_of_entities_text == 'vinvl_vgoi_entities':
            entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/vgcocooiobjects_v1_class2ind.json', not args.disable_all_entities)
            if args.prompt_ensemble:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vgoi_embeddings_{clip_name}_with_ensemble.pickle')
            else:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vgoi_embeddings_{clip_name}.pickle')
        else:
            print('The entities text should be input correctly!')
            return
    else:
        entities_text = None
        texts_embeddings = None
    
    # loading model
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, gpt_type = args.language_model)
    # 先加载到CPU，避免显存不足，然后再移动到GPU
    model.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict = False)
    model.to(device)
    encoder, preprocess = clip.load(args.clip_model, device = device)

    images_list = os.listdir(args.image_path)
    images_list.sort()
    images_list = [os.path.join(args.image_path, image) for image in images_list]

    predicts = []
    for _, im_path in tqdm(enumerate(images_list)):
        try:
            image = preprocess(Image.open(im_path)).unsqueeze(dim = 0).to(device)
        except:
            continue
        image_features = encoder.encode_image(image).float()  # (1, clip_hidden_size)
        image_features /= image_features.norm(2, dim = -1, keepdim = True)
        continuous_embeddings = model.mapping_network(image_features).view(-1, args.continuous_prompt_length, model.gpt_hidden_size)
        
        # ========== 实体检测（替换部分） ==========
        if args.using_hard_prompt:
            if args.use_memory and MEA_CAP_AVAILABLE and parser_model is not None:
                # ========== MeaCap Retrieve-then-Filter 方法 ==========
                try:
                    # 1. 计算图像特征（使用MeaCap的CLIP工具类）
                    batch_image_embeds = vl_model.compute_image_representation_from_image_path(im_path)
                    
                    # 2. 检索记忆库
                    if retrieve_on_CPU:
                        batch_image_embeds_cpu = batch_image_embeds.to('cpu')
                        memory_clip_embeddings_cpu = memory_clip_embeddings.to('cpu')
                        clip_score_cpu, clip_ref_cpu = vl_model.compute_image_text_similarity_via_embeddings(
                            batch_image_embeds_cpu,
                            memory_clip_embeddings_cpu
                        )
                        clip_score = clip_score_cpu.to(device)
                    else:
                        memory_clip_embeddings_gpu = memory_clip_embeddings.to(device)
                        clip_score, clip_ref = vl_model.compute_image_text_similarity_via_embeddings(
                            batch_image_embeds,
                            memory_clip_embeddings_gpu
                        )
                    
                    # 3. 选择Top-K记忆描述
                    select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
                    select_memory_captions = [memory_captions[id] for id in select_memory_ids]
                    
                    # 4. Retrieve-then-Filter提取概念
                    detected_objects = retrieve_concepts(
                        parser_model=parser_model,
                        parser_tokenizer=parser_tokenizer,
                        wte_model=wte_model,
                        select_memory_captions=select_memory_captions,
                        image_embeds=batch_image_embeds,
                        device=device
                    )
                except Exception as e:
                    # 回退到原始方法
                    if entities_text is not None and texts_embeddings is not None:
                        logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
                        detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
                        detected_objects = detected_objects[0]
                    else:
                        print(f"Error: Cannot retrieve concepts for image {im_path}")
                        continue
            else:
                # ========== ViECap 原始方法 ==========
                if entities_text is None or texts_embeddings is None:
                    print(f"Error: Entity vocabulary not loaded for image {im_path}")
                    continue
                logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
                detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold) # List[List[]], [[category1, category2, ...], []]
                detected_objects = detected_objects[0] # infering single image -> List[category1, category2, ...]
            
            # 后续处理完全相同
            discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)
            discrete_embeddings = model.word_embed(discrete_tokens)
            if args.only_hard_prompt:
                embeddings = discrete_embeddings
            elif args.soft_prompt_first:
                embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim = 1)
            else:
                embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim = 1)
        else:
            embeddings = continuous_embeddings
        
        # 文本生成（完全不变）
        if 'gpt' in args.language_model:
            if not args.using_greedy_search:
                sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt) # List[str]
                sentence = sentence[0] # selected top 1
            else:
                sentence = greedy_search(embeddings = embeddings, tokenizer = tokenizer, model = model.gpt)
        else:
            sentence = opt_search(prompts=args.text_prompt, embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt)
            sentence=sentence[0]
        
        _, im_name = os.path.split(im_path)
        predict = {}
        predict["image_name"] = im_name
        predict["prediction"] = sentence
        predicts.append(predict)
    
    outpath = os.path.join(args.image_path, 'predictions.json')
    with open(outpath, 'w') as outfile:
        json.dump(predicts, outfile, indent = 4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = 'cuda:0')
    parser.add_argument('--clip_model', default = 'ViT-B/32')
    parser.add_argument('--language_model', default = 'gpt2')
    parser.add_argument('--continuous_prompt_length', type = int, default = 10)
    parser.add_argument('--clip_project_length', type = int, default = 10)
    parser.add_argument('--temperature', type = float, default = 0.01)
    parser.add_argument('--top_k', type = int, default = 3)
    parser.add_argument('--threshold', type = float, default = 0.2)
    parser.add_argument('--disable_all_entities', action = 'store_true', default = False, help = 'whether to use entities with a single word only')
    parser.add_argument('--name_of_entities_text', default = 'vinvl_vgoi_entities', choices = ('visual_genome_entities', 'coco_entities', 'open_image_entities', 'vinvl_vg_entities', 'vinvl_vgoi_entities'))
    parser.add_argument('--prompt_ensemble', action = 'store_true', default = False)
    parser.add_argument('--weight_path', default = './checkpoints/train_coco/coco_prefix-0014.pt')
    parser.add_argument('--image_path', default = './images')
    parser.add_argument('--using_hard_prompt', action = 'store_true', default = False)
    parser.add_argument('--soft_prompt_first', action = 'store_true', default = False)
    parser.add_argument('--only_hard_prompt', action = 'store_true', default = False)
    parser.add_argument('--using_greedy_search', action = 'store_true', default = False, help = 'greedy search or beam search')
    parser.add_argument('--beam_width', type = int, default = 5, help = 'width of beam')
    parser.add_argument('--text_prompt', type = str, default = None)
    
    # MeaCap相关参数
    parser.add_argument('--use_memory', action = 'store_true', default = False, help = 'use MeaCap retrieve-then-filter module')
    parser.add_argument('--memory_id', type = str, default = 'coco', help = 'memory bank ID (coco, flickr30k, cc3m, ss1m)')
    parser.add_argument('--memory_caption_num', type = int, default = 5, help = 'number of memory captions to retrieve')
    parser.add_argument('--parser_checkpoint', type = str, default = 'lizhuang144/flan-t5-base-VG-factual-sg', help = 'scene graph parser checkpoint')
    parser.add_argument('--wte_model_path', type = str, default = 'sentence-transformers/all-MiniLM-L6-v2', help = 'SentenceBERT model path')
    parser.add_argument('--vl_model', type = str, default = 'openai/clip-vit-base-patch32', help = 'CLIP model for memory retrieval')
    
    args = parser.parse_args()
    print('args: {}\n'.format(vars(args)))

    main(args)

