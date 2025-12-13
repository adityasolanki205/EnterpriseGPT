import sys
print(sys.path)
try:
    import langchain
    print(f"LangChain Version: {langchain.__version__}")
    print(f"LangChain Path: {langchain.__file__}")
    from langchain import chains
    print("langchain.chains imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
