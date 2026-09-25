from langchain_community.document_loaders import YoutubeLoader 


def YTSloader(video_id,lang):
    YTtranscript_Loader = YoutubeLoader(video_id,False,lang,continue_on_failure=True)
    documents = YTtranscript_Loader.load()
    return documents