import React, { useRef, useState, useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

import { dharmaApi, DharmaAPIError } from '@/services/api';
import { useChatStore } from '@/store/chatStore';
import { useSettingsStore } from '@/store/settingsStore';
import { Colors, Fonts, Spacing, Radius } from '@/constants/theme';
import ModeSelector from '@/components/ModeSelector';
import MessageBubble from '@/components/MessageBubble';

const SUGGESTED_QUESTIONS = [
  'What does the Ramayana teach about dharma?',
  'How should I handle family conflicts according to scripture?',
  'Tell me the story of Hanuman crossing the ocean',
  'What is the meaning of karma in the Mahabharata?',
];

export default function ChatScreen() {
  const [input, setInput] = useState('');
  const flatRef = useRef<FlatList>(null);

  const messages = useChatStore((s) => s.messages);
  const isLoading = useChatStore((s) => s.isLoading);
  const mode = useChatStore((s) => s.mode);
  const language = useChatStore((s) => s.language);
  const setMode = useChatStore((s) => s.setMode);
  const setLoading = useChatStore((s) => s.setLoading);
  const addMessage = useChatStore((s) => s.addMessage);
  const updateMessage = useChatStore((s) => s.updateMessage);
  const getApiHistory = useChatStore((s) => s.getApiHistory);
  const clearHistory = useChatStore((s) => s.clearHistory);

  const haptic = useSettingsStore((s) => s.hapticFeedback);
  const settingsLanguage = useSettingsStore((s) => s.language);

  const activeLanguage = language || settingsLanguage;

  const scrollToBottom = useCallback(() => {
    setTimeout(() => flatRef.current?.scrollToEnd({ animated: true }), 100);
  }, []);

  const sendQuery = useCallback(
    async (queryText: string) => {
      const trimmed = queryText.trim();
      if (!trimmed || isLoading) return;

      setInput('');
      if (haptic) Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

      // Add user message
      addMessage({ role: 'user', content: trimmed });

      // Add placeholder assistant message
      const assistantId = addMessage({
        role: 'assistant',
        content: '',
        mode,
      });

      setLoading(true);
      scrollToBottom();

      try {
        const history = getApiHistory(10);
        const res = await dharmaApi.query({
          query: trimmed,
          mode,
          language: activeLanguage,
          history: history.slice(0, -1), // exclude the user msg just added
        });

        updateMessage(assistantId, {
          content: res.answer,
          sources: res.sources,
          query_id: res.query_id,
          mode: res.mode,
        });
      } catch (err) {
        const msg =
          err instanceof DharmaAPIError
            ? err.message
            : 'Could not reach the server. Please try again.';
        updateMessage(assistantId, {
          content: `⚠️ ${msg}`,
        });
      } finally {
        setLoading(false);
        scrollToBottom();
      }
    },
    [isLoading, mode, activeLanguage, haptic]
  );

  const handleFeedback = useCallback(
    (msgId: string, rating: 'up' | 'down') => {
      updateMessage(msgId, { feedback: rating });
    },
    []
  );

  const isEmpty = messages.length === 0;

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 88 : 0}
      >
        {/* Mode selector */}
        <ModeSelector selected={mode} onSelect={setMode} />

        {/* Messages */}
        {isEmpty ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>🕉  Ask anything about Dharma</Text>
            <Text style={styles.emptySubtitle}>
              Grounded in the Ramayana, Mahabharata, and Puranas
            </Text>
            <View style={styles.suggestions}>
              {SUGGESTED_QUESTIONS.map((q) => (
                <TouchableOpacity
                  key={q}
                  style={styles.suggestionChip}
                  onPress={() => sendQuery(q)}
                  activeOpacity={0.7}
                >
                  <Text style={styles.suggestionText}>{q}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        ) : (
          <FlatList
            ref={flatRef}
            data={messages}
            keyExtractor={(m) => m.id}
            renderItem={({ item }) => (
              <MessageBubble
                message={item}
                onFeedbackSubmit={handleFeedback}
              />
            )}
            contentContainerStyle={styles.messageList}
            onContentSizeChange={scrollToBottom}
          />
        )}

        {/* Loading indicator */}
        {isLoading && (
          <View style={styles.loadingRow}>
            <ActivityIndicator size="small" color={Colors.saffron} />
            <Text style={styles.loadingText}>Consulting the scriptures…</Text>
          </View>
        )}

        {/* Input bar */}
        <View style={styles.inputBar}>
          {messages.length > 0 && (
            <TouchableOpacity
              style={styles.clearBtn}
              onPress={() => {
                Alert.alert('Clear conversation?', '', [
                  { text: 'Cancel', style: 'cancel' },
                  { text: 'Clear', style: 'destructive', onPress: clearHistory },
                ]);
              }}
            >
              <Ionicons name="trash-outline" size={18} color={Colors.ash} />
            </TouchableOpacity>
          )}
          <TextInput
            style={styles.input}
            value={input}
            onChangeText={setInput}
            placeholder="Ask a question about Dharma…"
            placeholderTextColor={Colors.ash}
            multiline
            maxLength={1000}
            returnKeyType="send"
            onSubmitEditing={() => sendQuery(input)}
            blurOnSubmit={false}
          />
          <TouchableOpacity
            style={[
              styles.sendBtn,
              (!input.trim() || isLoading) && styles.sendBtnDisabled,
            ]}
            onPress={() => sendQuery(input)}
            disabled={!input.trim() || isLoading}
          >
            <Ionicons name="send" size={18} color={Colors.white} />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.darkBg,
  },
  flex: { flex: 1 },

  messageList: {
    paddingTop: Spacing.md,
    paddingBottom: Spacing.md,
  },

  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.xl,
  },
  emptyTitle: {
    fontSize: Fonts.sizes.xl,
    fontWeight: '700',
    color: Colors.cream,
    textAlign: 'center',
    marginBottom: Spacing.sm,
  },
  emptySubtitle: {
    fontSize: Fonts.sizes.sm,
    color: Colors.ash,
    textAlign: 'center',
    marginBottom: Spacing.xl,
  },
  suggestions: {
    width: '100%',
    gap: Spacing.sm,
  },
  suggestionChip: {
    backgroundColor: Colors.darkCard,
    borderRadius: Radius.md,
    padding: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.darkBorder,
  },
  suggestionText: {
    color: Colors.sandal,
    fontSize: Fonts.sizes.sm,
    lineHeight: 20,
  },

  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.sm,
  },
  loadingText: {
    color: Colors.ash,
    fontSize: Fonts.sizes.sm,
    fontStyle: 'italic',
  },

  inputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    backgroundColor: Colors.darkCard,
    borderTopWidth: 1,
    borderTopColor: Colors.darkBorder,
    gap: Spacing.sm,
  },
  clearBtn: {
    padding: 10,
    marginBottom: 2,
  },
  input: {
    flex: 1,
    backgroundColor: Colors.darkBg,
    borderRadius: Radius.md,
    paddingHorizontal: Spacing.md,
    paddingTop: 10,
    paddingBottom: 10,
    color: Colors.cream,
    fontSize: Fonts.sizes.md,
    maxHeight: 120,
    borderWidth: 1,
    borderColor: Colors.darkBorder,
  },
  sendBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.saffron,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 2,
  },
  sendBtnDisabled: {
    backgroundColor: Colors.darkBorder,
  },
});
